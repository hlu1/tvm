import click
from caffe2.proto import caffe2_pb2
from google.protobuf import text_format

@click.command()
@click.option('--net', type=click.Path())
def main(net):
    net = str(net)
    if net.endswith('.pb'):
        return
    assert net.endswith('.pbtxt')

    netdef = caffe2_pb2.NetDef()
    with open(net, 'r') as f:
        text_format.Merge(f.read(), netdef)

    with open(net[:-3], 'wb') as f:
        f.write(netdef.SerializeToString())

if __name__ == "__main__":
    main()