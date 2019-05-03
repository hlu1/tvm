from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time

import _pickle as pickle
import click
import numpy as np
import tvm
from tvm import autotvm, relay
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime

import dense
import batch_matmul

TARGETS = dict(
    skl="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu",
    local="llvm -mcpu=core-avx2",
)


def _extract_path(func_pkl):
    tokens = func_pkl.split("/")
    if tokens:
        return "".join(tokens[:-1]) + "/"
    return ""


@click.command()
@click.option("--func_pkl", type=click.Path())
@click.option("--params_pkl", type=click.Path())
@click.option("--inputs_pkl", type=click.Path())
@click.option("--num_iter", default=1000)
@click.option("--num_cycles", default=5)
@click.option("--opt_level", default=3)
@click.option("--autotvm_log", default="", type=click.Path())
@click.option("--tracker_port", default=9195)
@click.option("--device", type=click.Choice(TARGETS.keys()))
@click.option("--layerwise", is_flag=True, default=False)
def run(
    func_pkl,
    params_pkl,
    inputs_pkl,
    num_iter,
    num_cycles,
    opt_level,
    autotvm_log,
    tracker_port,
    device,
    layerwise,
):
    logging.basicConfig(level=logging.DEBUG)
    target = TARGETS[device]

    with open(func_pkl, "rb") as f:
        func = pickle.load(f)

    with open(params_pkl, "rb") as f:
        params = pickle.load(f)
        params = {
            name: tvm.nd.array(np.atleast_1d(np.random.randn(*shape)).astype(dtype))
            for name, (shape, dtype) in params.items()
        }

    with open(inputs_pkl, "rb") as f:
        inputs = pickle.load(f)
        inputs = {name: tvm.nd.array(v) for name, v in inputs.items()}

    if autotvm_log:
        with autotvm.apply_history_best(str(autotvm_log)):
            with tvm.target.create(target):
                with relay.build_config(opt_level=opt_level):
                    graph, lib, new_params = relay.build(func, target, params=params)
    else:
        with tvm.target.create(target):
            with relay.build_config(opt_level=opt_level):
                graph, lib, new_params = relay.build(func, target, params=params)

    with open(_extract_path(func_pkl) + "graph.json", "w") as f:
        f.write(graph)

    if device == "skl":
        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath("net.tar")
        with tvm.target.create(target):
            lib.export_library(lib_fname)
        tracker = tvm.rpc.connect_tracker("localhost", 9195)
        remote = tracker.request("skl")
        remote.upload(lib_fname)
        lib = remote.load_module("net.tar")
        ctx = remote.cpu(0)
    else:
        ctx = tvm.context(str(target), 0)

    if layerwise:
        module = debug_runtime.create(graph, lib, ctx)
    else:
        module = graph_runtime.create(graph, lib, ctx)

    logging.debug(graph)

    for k, v in sorted(new_params.items()):
        logging.debug("{}: {}".format(k, v.shape))

    module.set_input(**inputs)
    module.set_input(**new_params)
    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for _ in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)
        time.sleep(1)
    if layerwise:
        module.run()


if __name__ == "__main__":
    run()
