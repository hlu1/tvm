#!/usr/bin/env python
# coding: utf-8

from caffe2.proto import caffe2_pb2
from caffe2.proto import metanet_pb2
from caffe2.python.predictor_constants import predictor_constants
from caffe2.python import workspace

from c2_models import models, load_caffe2_netdef
from export_model import save_model

import click

@click.command()
@click.option('--model',
              type=click.Choice(['seg_detect', 'seg_track', 'seg', 'seg_le', 'seg_le_detect', 'seg_le_track',]),
              required=True)
def run(model):
    seg_model = models[model]
    init, predict = load_caffe2_netdef(seg_model.init_net, seg_model.predict_net)

    for op in predict.op:
        for arg in op.arg:
            if arg.name == "algo":
                print(arg.s)
                arg.s = str.encode("WINOGRAD")
    print(predict)
    save_model(seg_model.predict_net + ".fp32.pb", predict)

if __name__ == "__main__":
    run()
