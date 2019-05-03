from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import click

from caffe2.proto import caffe2_pb2
from c2_frontend import from_caffe2
from caffe2.python.predictor import mobile_exporter
from caffe2.python import core, dyndep, workspace
import numpy as np
import tvm
import tvm.rpc
from tvm import relay
from tvm.contrib import graph_runtime

dyndep.InitOpsLibrary("//caffe2/caffe2/fb/tvm:c2_frontend")

skl_target = tvm.target.create("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu")
local_target = tvm.target.create("llvm -mcpu=core-avx2")


def _get_caffe2_output(init_net, input_init_net, predict_net):
    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    workspace.RunNetOnce(input_init_net)
    workspace.RunNetOnce(predict_net)

    outputs = []
    for output in predict_net.external_output:
        outputs.append(workspace.FetchBlob(output))
    return outputs





@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--input_init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--tracker_port", default=9190)
@click.option("--opt_level", default=3)
@click.option("--device", type=click.Choice(["skl", "local"]), required=True)
@click.option("--verbose", is_flag=True, default=False)
def run(
    init_net,
    input_init_net,
    pred_net,
    tracker_port,
    opt_level,
    device,
    verbose,
):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)

    target = dict(skl=skl_target, local=local_target)[device]

    with open(init_net, "rb") as f:
        init_net = caffe2_pb2.NetDef()
        init_net.ParseFromString(f.read())

    with open(input_init_net, "rb") as f:
        input_init_net = caffe2_pb2.NetDef()
        input_init_net.ParseFromString(f.read())

    with open(pred_net, "rb") as f:
        pred_net = caffe2_pb2.NetDef()
        pred_net.ParseFromString(f.read())

    # get device
    if device == "local":
        remote = tvm.rpc.LocalSession()
    else:
        tracker = tvm.rpc.connect_tracker("localhost", tracker_port)
        remote = tracker.request(device)
    ctx = remote.cpu(0)

    def _get_tvm_output(func, inputs, params, output_sizes, opt_level=3):
        with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build(func, target, params=params)

        tmp = tvm.contrib.util.tempdir()
        fname = "net.tar"
        lib_fname = tmp.relpath(fname)

        with tvm.target.create(target):
            lib.export_library(lib_fname)

        remote.upload(lib_fname)
        rlib = remote.load_module(fname)

        module = graph_runtime.create(graph, rlib, ctx)
        module.set_input(**inputs)
        module.set_input(**params)
        module.run()

        outputs = []
        for i, output_size in enumerate(output_sizes):
            outputs.append(module.get_output(i, tvm.nd.empty(output_size, ctx=ctx)))
        return outputs

    func = from_caffe2(init_net, input_init_net, pred_net)

    workspace.ResetWorkspace()
    workspace.RunNetOnce(input_init_net)
    inputs = {name: workspace.FetchBlob(name) for name in workspace.Blobs()}

    workspace.ResetWorkspace()
    workspace.RunNetOnce(init_net)
    params = {name: workspace.FetchBlob(name) for name in workspace.Blobs()}

    c2_outputs = _get_caffe2_output(init_net, input_init_net, pred_net)


    output_sizes = [out.shape for out in c2_outputs]

    # run tvm
    import ipdb; ipdb.set_trace()
    tvm_outputs = [out.asnumpy() for out in _get_tvm_output(func, inputs, params, output_sizes))

    for idx in range(len(output_sizes)):
        tvm.testing.assert_allclose(c2_outputs[idx], tvm_outputs[idx], rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run()
