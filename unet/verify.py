from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import click
import nnvm
import numpy as np
import tvm
import tvm.contrib.util
import tvm.rpc
from caffe2.proto import caffe2_pb2, metanet_pb2
from caffe2.python import workspace
from tvm import autotvm, relay
from tvm.contrib import graph_runtime


skl_target = tvm.target.create("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu")
local_target = tvm.target.create("llvm -mcpu=core-avx2")
rpi_target = tvm.target.arm_cpu("rasp3b")


def _get_caffe2_output(init_net, predict_net, input):
    workspace.RunNetOnce(init_net)

    input_blob = predict_net.op[0].input[0]
    workspace.FeedBlob(input_blob, input)
    workspace.RunNetOnce(predict_net)

    output_blob = predict_net.external_output[0]
    c2_output = workspace.FetchBlob(output_blob)
    return c2_output


@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--input_size", type=(int, int, int, int), default=(1, 3, 192, 192))
@click.option(
    "--input_dtype", type=click.Choice(["float32", "uint8"]), default="float32"
)
@click.option("--output_size", type=(int, int, int, int), default=(1, 1, 192, 192))
@click.option("--autotvm_log", type=click.Path())
@click.option("--frontend", type=click.Choice(["nnvm", "relay"]), default="nnvm")
@click.option("--tracker_port", default=9190)
@click.option("--opt_level", default=3)
@click.option("--device", type=click.Choice(["skl", "rpi", "local"]), required=True)
@click.option("--verbose", is_flag=True, default=False)
def run(
    init_net,
    pred_net,
    input_size,
    input_dtype,
    output_size,
    autotvm_log,
    frontend,
    tracker_port,
    opt_level,
    device,
    verbose,
):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)

    target = dict(skl=skl_target, rpi=rpi_target, local=local_target)[device]

    try:
        with open(init_net, "rb") as f:
            init_meta_net = metanet_pb2.MetaNetDef()
            init_meta_net.ParseFromString(f.read())
        with open(pred_net, "rb") as f:
            pred_meta_net = metanet_pb2.MetaNetDef()
            pred_meta_net.ParseFromString(f.read())
        init_net = init_meta_net.nets[0].value
        pred_net = pred_meta_net.nets[0].value
    except Exception as e:
        logging.exception(e)
        logging.info(
            "Failed to read inputs as MetaNetDefs, read inputs as NetDefs instead"
        )
        with open(init_net, "rb") as f:
            init_net = caffe2_pb2.NetDef()
            init_net.ParseFromString(f.read())
        with open(pred_net, "rb") as f:
            pred_net = caffe2_pb2.NetDef()
            pred_net.ParseFromString(f.read())

    # run caffe2
    data = np.random.uniform(size=(input_size)).astype(input_dtype)

    input_name = pred_net.external_input[0]
    c2_out = _get_caffe2_output(init_net, pred_net, data)

    # get device
    if device == "local":
        remote = tvm.rpc.LocalSession()
    else:
        tracker = tvm.rpc.connect_tracker("localhost", tracker_port)
        remote = tracker.request(device)
    ctx = remote.cpu(0)

    # run tvm with nnvm
    def _compile_with_nnvm(init_net, pred_net):
        sym, params = nnvm.frontend.from_caffe2(init_net, pred_net)
        logging.debug(sym.debug_str())

        with autotvm.apply_history_best(str(autotvm_log)):
            with nnvm.compiler.build_config(opt_level=opt_level):
                with tvm.build_config(partition_const_loop=True):
                    graph, lib, params = nnvm.compiler.build_module.build(
                        sym, target, shape={input_name: input_size}, params=params
                    )
                    logging.debug(graph.symbol().debug_str())
        return graph, lib, params

    def _compile_with_relay(init_net, pred_net):
        shape_dict = {input_name: data.shape}
        dtype_dict = {input_name: data.dtype}
        func, params = relay.frontend.from_caffe2(
            init_net, pred_net, shape_dict, dtype_dict
        )

        with autotvm.apply_history_best(str(autotvm_log)):
            with relay.build_config(opt_level=opt_level):
                graph, lib, params = relay.build(func, target, params=params)
                logging.debug(graph)
        return graph, lib, params

    def _run_tvm_graph(graph, lib, params, fname):
        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath(fname)

        with tvm.target.create(target):
            lib.export_library(lib_fname)

        remote.upload(lib_fname)
        rlib = remote.load_module(fname)

        module = graph_runtime.create(graph, rlib, ctx)
        if data is not None:
            module.set_input(input_name, tvm.nd.array(data.astype("float32")))
        module.set_input(**params)
        module.run()

        out = module.get_output(0, tvm.nd.empty(output_size, ctx=ctx))
        return module, [out]

    if frontend == "nnvm":
        graph, lib, params = _compile_with_nnvm(init_net, pred_net)
    else:
        graph, lib, params = _compile_with_relay(init_net, pred_net)

    _, outs = _run_tvm_graph(graph, lib, params, "net.tar")
    tvm_out = outs[0].asnumpy()

    tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    run()
