from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import shutil
import tempfile

import click
import nnvm
import tvm
import tvm.contrib.util
from caffe2.proto import caffe2_pb2, metanet_pb2
from nnvm.frontend import from_caffe2
from tvm import autotvm

import build_module


skl_target = tvm.target.create("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu")
local_target = tvm.target.create("llvm -mcpu=core-avx2")
rpi_target = tvm.target.arm_cpu("rasp3b")
android_target = tvm.target.create(
    "llvm -device=arm_cpu -target=armv7-none-linux-androideabi -mfloat-abi=soft"
)


def _save_graph(graph, fname):
    g = graph.json()
    with open(fname, "w") as f:
        f.write(g)


def _dump_params(params):
    for k, v in params.items():
        logging.info(k, v.dtype, v.shape)


def _save_lib(lib, fname, target, device):
    assert lib
    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath(fname)
    fcompile = None
    if device == "android":
        from tvm.contrib import ndk

        fcompile = ndk.create_shared
    with tvm.target.create(target):
        lib.export_library(lib_fname, fcompile)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        shutil.copyfile(lib_fname, f.name)
    return f.name


def _add_arg(net, name, s):
    arg = net.arg.add()
    arg.name = name
    arg.s = s


def _pack_model(
    predict,
    graph,
    lib,
    pre_graph=None,
    lib_pregraph=None,
    device="llvm",
    arm_arch="armv7",
):
    suffix = "_" + arm_arch if device == "android" else ""
    if pre_graph is not None:
        assert isinstance(pre_graph, str)
        _add_arg(predict, "tvm_pregraph" + suffix, str.encode(pre_graph))

    if lib_pregraph is not None:
        with open(lib_pregraph, "rb") as f:
            _add_arg(predict, "tvm_lib_pregraph" + suffix, f.read())

    assert isinstance(graph, str)
    _add_arg(predict, "tvm_graph" + suffix, str.encode(graph))

    with open(lib, "rb") as f:
        _add_arg(predict, "tvm_lib" + suffix, f.read())
    return predict


def _save_model(output_file, net):
    with open(output_file, "wb") as f:
        f.write(net.SerializeToString())
        print("Exported model saved at:", output_file)


@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--input_size1", type=(int, int, int, int), default=(1, 3, 192, 192))
@click.option("--input_size2", type=(int, int, int, int), default=(1, 4, 192, 192))
@click.option("--output", type=click.Path())
@click.option("--autotvm_log", type=click.Path())
@click.option("--opt_level", default=3)
@click.option(
    "--device", type=click.Choice(["skl", "rpi", "local", "android"]), required=True
)
# Todo: add arm64 support
@click.option("--arm_arch", type=click.Choice(["armv7", "arm64"]), default="armv7")
@click.option("--verbose", is_flag=True, default=False)
def main(
    init_net,
    pred_net,
    input_size1,
    input_size2,
    output,
    autotvm_log,
    opt_level,
    device,
    arm_arch,
    verbose,
):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)

    target = dict(
        skl=skl_target, rpi=rpi_target, local=local_target, android=android_target
    )[device]

    def compile(net, sym, params, input_name, data_shape):
        with autotvm.apply_history_best(str(autotvm_log)):
            with nnvm.compiler.build_config(opt_level=opt_level):
                with tvm.build_config(partition_const_loop=True):
                    pre_graph, graph, lib, _ = build_module.build(
                        sym, target, shape={input_name: data_shape}, params=params
                    )
                    lib_f = _save_lib(
                        lib,
                        "net.so" if device == "android" else "net.tar",
                        target,
                        device,
                    )
                    logging.debug(graph.symbol().debug_str())

            if pre_graph:
                with nnvm.compiler.build_config(opt_level=0):
                    shape = {k: v.shape for k, v in params.items()}
                    dtype = {k: v.dtype for k, v in params.items()}
                    _, pre_graph, lib_pregraph, _ = build_module.build(
                        pre_graph, target, shape, dtype
                    )
                    lib_pregraph_f = _save_lib(
                        lib_pregraph,
                        "pre_graph.so" if device == "android" else "pre_graph.tar",
                        target,
                        device,
                    )
                    logging.debug(pre_graph.symbol().debug_str())

        return _pack_model(
            net, graph.json(), lib_f, pre_graph.json(), lib_pregraph_f, device, arm_arch
        )

    try:
        with open(init_net, "rb") as f:
            init_meta_net = metanet_pb2.MetaNetDef()
            init_meta_net.ParseFromString(f.read())
        with open(pred_net, "rb") as f:
            pred_meta_net = metanet_pb2.MetaNetDef()
            pred_meta_net.ParseFromString(f.read())

        num_models = len(pred_meta_net.nets)
        num_models = len(pred_meta_net.nets)
        assert num_models <= 2
        if num_models == 2:
            input_shapes = [input_size1, input_size2]
        else:
            input_shapes = [input_size1]
        for index in range(num_models):
            init = init_meta_net.nets[index].value
            pred = pred_meta_net.nets[index].value
            input_name = pred.external_input[0]
            sym, params = from_caffe2(init, pred)
            init = compile(init, sym, params, input_name, input_shapes[index])
        _save_model(output, init_meta_net)
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
        input_name = pred_net.external_input[0]
        sym, params = from_caffe2(init_net, pred_net)
        init_net = compile(init_net, sym, params, input_name, input_size1)
        _save_model(output, init_net)


if __name__ == "__main__":
    main()
