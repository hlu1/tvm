import click
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
import copy
import logging
from utils import set_batch_size
logging.basicConfig(level=logging.DEBUG)


def print_all_blobs(ws):
    for k in ws.Blobs():
        b = ws.FetchBlob(k)
        logging.info("%s: %s", k, b.shape)


@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--input_init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--num_iter", default=10000)
@click.option("--num_cycles", default=5)
@click.option("--batch_size", type=(int, int), default=(20, 20))
@click.option("--layerwise", is_flag=True, default=False)
def main(
    init_net, input_init_net, pred_net, num_iter, num_cycles, batch_size, layerwise
):
    with open(init_net, "rb") as f:
        init_net = caffe2_pb2.NetDef()  # weights
        init_net.ParseFromString(f.read())

    with open(input_init_net, "rb") as f:
        input_init_net = caffe2_pb2.NetDef()
        input_init_net.ParseFromString(f.read())

    with open(pred_net, "rb") as f:
        pred_net = caffe2_pb2.NetDef()
        pred_net.ParseFromString(f.read())

    fake_init_net = copy.copy(init_net)
    fake_init_net.op.extend([op for op in input_init_net.op])
    workspace.RunNetOnce(fake_init_net)

    print_all_blobs(workspace)

    # Set batch_size
    ws = workspace.C.Workspace()
    ws.run(input_init_net)
    input_dict = {name: ws.fetch_blob(name) for name in ws.blobs.keys()}
    input_dict = set_batch_size(input_dict, batch_size)
    for k, v in input_dict.items():
        workspace.FeedBlob(k, v)

    pred_net.name = "benchmark"
    workspace.CreateNet(pred_net, True)
    for _ in range(num_cycles):
        workspace.BenchmarkNet("benchmark", num_iter, num_iter, layerwise)


if __name__ == "__main__":
    main()
