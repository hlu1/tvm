from tvm import autotvm
from tvm.contrib import graph_runtime
import tvm.contrib.util
import tvm.rpc
import click
import logging
import nnvm
import nnvm.compiler
import numpy as np
import tvm
import shutil

from c2_models import get_c2_symbol, get_c2_model, get_caffe2_output, load_caffe2_netdef
from build import build

# skl_target = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')
# local_target = tvm.target.create('llvm -mcpu=core-avx2')
# rpi_target = tvm.target.arm_cpu("rasp3b")




def get_args(net):
    args = {}
    for arg in net.arg:
        args[arg.name] = arg.s
    return args

def get_arg(args, key):
    if key in args:
        return args[key]
    else:
        return None

def load_graph(s):
    if s is None:
        return None
    import json
    graph = json.loads(s)
    print(graph)
    return graph

def load_lib(s, device):
    if s is None:
        return None
    fname = 'lib.so' if device == 'android' else 'lib.o'
    tmp = tvm.contrib.util.tempdir()
    lib_fname = tmp.relpath(fname)
    with open(lib_fname, 'wb') as f:
        f.write(s)
    return lib_fname

def load_packed_model(init, predict, device):
    init, predict = load_caffe2_netdef(init, predict)

    args = get_args(predict)

    pregraph = load_graph(get_arg(args, 'tvm_pregraph'))
    graph = load_graph(get_arg(args, 'tvm_ir'))

    lib_pregraph = load_lib(get_arg(args, 'lib_pregraph'), device)
    lib = load_lib(get_arg(args, 'tvm_lib'), device)

    return pregraph, graph, lib_pregraph, lib

@click.command()
@click.option('--model',
              type=click.Choice(['seg_detect', 'seg_track']),
              required=True)
@click.option('--init', required=True, type=str)
@click.option('--predict', required=True, type=str)
@click.option('--device', type=click.Choice(["skl", "rpi", "local", "android"]), required=True)
@click.option('--verbose', is_flag=True, default=False)
def run(model,
        init,
        predict,
        device,
        verbose):
    logging.basicConfig(level=logging.INFO if not verbose else logging.DEBUG)

    # get params
    sym, params, image_shape, output_shape = get_c2_symbol(model)
    data_shape = tuple([1] + list(image_shape))
    input_name = '0'

    pregraph, graph, lib_pregraph, lib = load_packed_model(init, predict, device)

    lib = tvm.module.load(lib)
    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, lib, ctx)

    dtype = 'float32'
    input_data = np.random.uniform(size=data_shape).astype(dtype)

    m.set_input(input_name, tvm.nd.array(input_data.astype(input_data.dtype)))
    # m.set_input(**params)

    # execute
    m.run()

    ftimer = m.module.time_evaluator("run", ctx, 5)
    for i in range(1):
        prof_res = ftimer()
        # time.sleep(1)

    for i in range(2):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)

if __name__ == "__main__":
    run()
