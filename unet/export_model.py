from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import shutil
import tempfile

import build_module
import click
import nnvm
import tvm
import tvm.contrib.util
from caffe2.proto import caffe2_pb2, metanet_pb2
from nnvm.frontend import from_caffe2
from tvm import autotvm


class C2Exporter(object):
    def __init__(
        self,
        init_net,
        pred_net,
        input_sizes,
        output,
        autotvm_log,
        opt_level=3,
        device="local",
        arch=None,
    ):
        # read input model
        self.init_net_path = init_net
        self.pred_net_path = pred_net
        self.is_meta_netdef = False
        try:
            with open(init_net, "rb") as f:
                self.init_net = metanet_pb2.MetaNetDef()
                self.init_net.ParseFromString(f.read())
            with open(pred_net, "rb") as f:
                self.pred_net = metanet_pb2.MetaNetDef()
                self.pred_net.ParseFromString(f.read())
                print(self.pred_net)
            self.is_meta_netdef = True
        except Exception as e:
            logging.exception(e)
            logging.info(
                "Failed to read inputs as MetaNetDefs, read inputs as NetDefs instead"
            )
            with open(init_net, "rb") as f:
                self.init_net = caffe2_pb2.NetDef()
                self.init_net.ParseFromString(f.read())
            with open(pred_net, "rb") as f:
                self.pred_net = caffe2_pb2.NetDef()
                self.pred_net.ParseFromString(f.read())
        self.input_sizes = input_sizes if input_sizes else []
        self.output = output
        self.autotvm_log = autotvm_log
        self.opt_level = opt_level
        self.device = device
        self.arch = arch

        skl_target = tvm.target.create(
            "llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu"
        )
        local_target = tvm.target.create("llvm -mcpu=core-avx2")
        rpi_target = tvm.target.arm_cpu("rasp3b")
        android_armv7_target = tvm.target.create(
            "llvm -device=arm_cpu -target=armv7-none-linux-androideabi -mattr=+neon -mfloat-abi=soft"
        )
        android_arm64_target = tvm.target.create(
            "llvm -device=arm_cpu -target=arm64-none-linux-android -mattr=+neon -mfloat-abi=soft"
        )
        self.targets = {
            "skl": skl_target,
            "local": local_target,
            "rpi": rpi_target,
            "android_armv7": android_armv7_target,
            "android_arm64": android_arm64_target,
        }

    def _save_graph(self, graph, fname):
        g = graph.json()
        with open(fname, "w") as f:
            f.write(g)

    def _dump_params(self, params):
        for k, v in params.items():
            logging.info(k, v.dtype, v.shape)

    def _add_arg(self, net, name, s):
        arg = net.arg.add()
        arg.name = name
        arg.s = s

    def save_model(self, output, net):
        if not output.endswith(".pb"):
            output = output + "/tvm.pb"
        with open(output, "wb") as f:
            f.write(net.SerializeToString())
            logging.info("Exported model saved at: %s" % output)

    def set_version(self, version):
        if self.is_meta_netdef:
            self.init_net.modelInfo.version = str(version)
            self.pred_net.modelInfo.version = str(version)

    def _save_lib(self, lib, fname, target, arch):
        tmp = tvm.contrib.util.tempdir()
        lib_fname = tmp.relpath(fname)
        fcompile = None
        if self.device == "android":
            if arch == "armv7":
                os.environ[
                    "TVM_NDK_CC"
                ] = "/opt/android-toolchain-arm/bin/arm-linux-androideabi-clang++"
            elif arch == "arm64":
                os.environ[
                    "TVM_NDK_CC"
                ] = "/opt/android-toolchain-arm64/bin/aarch64-linux-android-clang++"
            else:
                raise Exception("unsupported arch: %s" % arch)
            from tvm.contrib import ndk

            fcompile = ndk.create_shared

        with tvm.target.create(target):
            lib.export_library(lib_fname, fcompile)
        with tempfile.NamedTemporaryFile(delete=False) as f:
            shutil.copyfile(lib_fname, f.name)
        return f.name

    def _convert_args(self, pre_graph, lib_pregraph, graph, lib, target, arch=None):
        suffix = "_" + arch if arch else ""
        args = {}
        fmt = ".so" if self.device == "android" else ".tar"
        if pre_graph:
            assert isinstance(pre_graph, nnvm.graph.Graph)
            args["tvm_pregraph" + suffix] = str.encode(pre_graph.json())

        if lib_pregraph:
            f_name = self._save_lib(lib_pregraph, "pre_graph" + fmt, target, arch)
            with open(f_name, "rb") as f:
                args["tvm_lib_pregraph" + suffix] = f.read()

        assert isinstance(graph, nnvm.graph.Graph)
        args["tvm_graph" + suffix] = str.encode(graph.json())

        assert lib
        f_name = self._save_lib(lib, "graph" + fmt, target, arch)
        with open(f_name, "rb") as f:
            args["tvm_lib" + suffix] = f.read()

        return args

    def _pack_model(self, net, args):
        for k, v in args.items():
            self._add_arg(net, k, v)
        return net

    def build(self, sym, params, target, input_name, data_shape):
        """
        Given a c2 model, convert into nnvm graph, run tvm build_module.build, and return
        the generated pre_graph, lib_pregraph.so, graph, lib_graph.so
        """
        with autotvm.apply_history_best(str(self.autotvm_log)):
            with nnvm.compiler.build_config(opt_level=self.opt_level):
                with tvm.build_config(partition_const_loop=True):
                    pre_graph, graph, lib, _ = build_module.build(
                        sym, target, shape={input_name: data_shape}, params=params
                    )
                    logging.debug(graph.symbol().debug_str())

            if pre_graph:
                with nnvm.compiler.build_config(opt_level=0):
                    shape = {k: v.shape for k, v in params.items()}
                    dtype = {k: v.dtype for k, v in params.items()}
                    _, pre_graph, lib_pregraph, _ = build_module.build(
                        pre_graph, target, shape, dtype
                    )
                    logging.debug(pre_graph.symbol().debug_str())
            else:
                pre_graph, lib_pregraph = None, None
        return (pre_graph, lib_pregraph, graph, lib)

    def export(self, version=None):
        """
        for every sub model in the model (when model is meta_netdef):
            for every arm arch (when is android):
                pre_graph, lib_pregraph.so, graph, lib_graph.so = build(model)
                save all .so onto disk
                add all four objects to init_net
        save init_net
        """
        if self.is_meta_netdef:
            self.export_meta_netdef(version)
        else:
            self.export_netdef()

    def process_net(self, init_net, pred_net, input_name, input_size):
        sym, params = from_caffe2(init_net, pred_net)
        if self.device == "android":
            if self.arch == "all":
                for arch in ["armv7", "arm64"]:
                    device = self.device + "_" + arch
                    target = self.targets[device]
                    pre_graph, lib_pregraph, graph, lib = self.build(
                        sym, params, target, input_name, input_size
                    )
                    args = self._convert_args(
                        pre_graph, lib_pregraph, graph, lib, target, arch
                    )
                    self._pack_model(init_net, args)
            else:
                assert self.arch in ["armv7", "arm64"]
                device = self.device + "_" + self.arch
                target = self.targets[device]
                pre_graph, lib_pregraph, graph, lib = self.build(
                    sym, params, target, input_name, input_size
                )
                args = self._convert_args(
                    pre_graph, lib_pregraph, graph, lib, target, self.arch
                )
                self._pack_model(init_net, args)
        else:
            target = self.targets[self.device]
            pre_graph, lib_pregraph, graph, lib = self.build(
                sym, params, target, input_name, input_size
            )
            args = self._convert_args(pre_graph, lib_pregraph, graph, lib, target)
            self._pack_model(init_net, args)
        return init_net, pred_net

    def export_netdef(self):
        assert not self.is_meta_netdef
        input_name = self.pred_net.external_input[0]
        assert len(self.input_sizes) == 1
        self.process_net(
            self.init_net, self.pred_net, input_name, self.input_sizes[0]
        )
        # save init_net to output
        self.save_model(self.output, self.init_net)

    def export_meta_netdef(self, version=None):
        assert self.is_meta_netdef
        num_models = len(self.pred_net.nets)
        assert len(self.input_sizes) == num_models
        for index in range(num_models):
            init_net = self.init_net.nets[index].value
            pred_net = self.pred_net.nets[index].value
            input_name = pred_net.external_input[0]
            self.process_net(init_net, pred_net, input_name, self.input_sizes[index])
        # save models
        if version:
            self.set_version(version)
            if self.output.endswith(".pb"):
                is_root_dir = self.output.startswith("/")
                tokens = self.output.split("/")
                self.output = (
                    ("/" + "/".join(tokens[:-1]))
                    if is_root_dir
                    else "/".join(tokens[:-1])
                )
                # handle empty case
                if not self.output:
                    self.output = "."
            self.save_model(
                self.output + "/" + self.init_net_path.split("/")[-1], self.init_net
            )
            self.save_model(
                self.output + "/" + self.pred_net_path.split("/")[-1], self.pred_net
            )
        else:
            self.save_model(self.output, self.init_net)


@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--input_size1", nargs=4, type=int, default=(1, 3, 192, 192))
@click.option("--input_size2", nargs=4, type=int)
@click.option("--output", type=click.Path())
@click.option("--version", type=str, default=None)
@click.option("--autotvm_log", type=click.Path())
@click.option("--opt_level", default=3)
@click.option(
    "--device", type=click.Choice(["skl", "rpi", "local", "android"]), required=True
)
@click.option("--arch", type=click.Choice(["armv7", "arm64", "all"]), default=None)
@click.option("--verbose", is_flag=True, default=False)
def main(
    init_net,
    pred_net,
    input_size1,
    input_size2,
    output,
    version,
    autotvm_log,
    opt_level,
    device,
    arch,
    verbose,
):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)
    input_sizes = [input_size1, input_size2] if input_size2 else [input_size1]
    exporter = C2Exporter(
        init_net, pred_net, input_sizes, output, autotvm_log, opt_level, device, arch
    )
    exporter.export(version)


if __name__ == "__main__":
    main()
