import click
import json
from collections import namedtuple
from c2_models import models, get_c2_model, get_c2_symbol
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace

# path = "/Users/hlu/github/tvm/unet/"


def add_arg(net, name, s):
    arg = net.arg.add()
    arg.name = name
    arg.s = s

def pack_model(model, path, graph, lib, pre_graph=None, lib_pregraph=None):
    c2_model = get_c2_model(model)
    _, predict = c2_model.init_net, c2_model.predict_net

    with open(path + graph, 'rb') as f:
        add_arg(predict, 'tvm_ir', f.read())

    with open(path + lib, 'rb') as f:
        add_arg(predict, 'tvm_lib', f.read())

    if pre_graph is not None:
        with open(path + pre_graph, 'rb') as f:
            add_arg(predict, 'tvm_pregraph', f.read())

    if lib_pregraph is not None:
        with open(path + lib_pregraph, 'rb') as f:
            add_arg(predict, 'lib_pregraph', f.read())

    with open(path + model + '_packed.pb', 'wb') as f:
        f.write(predict.SerializeToString())

@click.command()
@click.option('--model',
              type=click.Choice(['seg_detect', 'seg_track']),
              required=True)
@click.option('--path', required=True, type=str)
@click.option('--graph', required=True, type=str)
@click.option('--lib', required=True, type=str)
@click.option('--pre_graph', default=None, type=str)
@click.option('--lib_pregraph', default=None, type=str)
def run(model, path, graph, lib, pre_graph, lib_pregraph):
    pack_model(model, path, graph, lib, pre_graph, lib_pregraph)


if __name__ == '__main__':
    run()
