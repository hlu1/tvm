from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import time

import click
import pickle
import tvm
from caffe2.proto import caffe2_pb2
from tvm import relay
from tvm.contrib import graph_runtime, tar as _tar
from tvm.contrib.debugger import debug_runtime


TARGETS = dict(
    skl="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu",
    dev="llvm -mcpu=skylake -target=x86_64-linux-gnu",
    mac="llvm -mcpu=core-avx2",
)


def _export_library(path_obj, file_name):
    files = [path_obj]
    assert file_name.endswith(".tar")
    fcompile = _tar.tar
    fcompile(file_name, files)


@click.command()
# input model is c2 protobuf
@click.option("--init_net", type=click.Path())
@click.option("--input_init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
#input model is pkl
@click.option("--func_pkl", type=click.Path())
@click.option("--params_pkl", type=click.Path())
@click.option("--inputs_pkl", type=click.Path())
# input model is tvm compiled code and graph
@click.option("--object_code", type=click.Path())
@click.option("--graph", type=click.Path())
# args
@click.option("--num_iter", default=1000)
@click.option("--num_cycles", default=5)
@click.option("--layerwise", is_flag=True, default=False)
@click.option("--tracker_port", default=9195)
@click.option("--device", type=click.Choice(TARGETS.keys()), required=True)
@click.option("--verbose", is_flag=True, default=False)
def run(
    init_net,
    input_init_net,
    pred_net,
    func_pkl,
    params_pkl,
    inputs_pkl,
    object_code,
    graph,
    num_iter,
    num_cycles,
    layerwise,
    tracker_port,
    device,
    verbose,
):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)
    target = TARGETS[device]

    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath("net.tar")

    if init_net and input_init_net and pred_net:
        with open(init_net, "rb") as f:
            init_net = caffe2_pb2.NetDef()
            init_net.ParseFromString(f.read())

        with open(input_init_net, "rb") as f:
            input_init_net = caffe2_pb2.NetDef()
            input_init_net.ParseFromString(f.read())

        with open(pred_net, "rb") as f:
            pred_net = caffe2_pb2.NetDef()
            pred_net.ParseFromString(f.read())

        func, params = from_caffe2(init_net, input_init_net, pred_net)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target, params=params)

        with tvm.target.create(target):
            lib.export_library(lib_fname)

    elif func_pkl and params_pkl and inputs_pkl:
        with open(func_pkl, "rb") as f:
            func = pickle.load(f)

        with open(params_pkl, "rb") as f:
            params = pickle.load(f)

        with open(inputs_pkl, "rb") as f:
            inputs = pickle.load(f)

        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target, params=params)

        with tvm.target.create(target):
            lib.export_library(lib_fname)

    elif object_code and graph:
        with open(graph, "r") as f:
            graph = f.read()
        _export_library(object_code, lib_fname)
    else:
        raise Exception("Bad arguments")

    if device == "skl":
        tracker = tvm.rpc.connect_tracker("localhost", 9195)
        remote = tracker.request("skl")
    else:
        remote = tvm.rpc.LocalSession()

    remote.upload(lib_fname)
    lib = remote.load_module("net.tar")
    ctx = remote.cpu(0)

    if layerwise:
        module = debug_runtime.create(graph, lib, ctx)
    else:
        module = graph_runtime.create(graph, lib, ctx)

    logging.debug(graph)

    ftimer = module.module.time_evaluator("run", ctx, num_iter)
    for _ in range(num_cycles):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)
        time.sleep(1)
    if layerwise:
        module.run()


if __name__ == "__main__":
    run()
