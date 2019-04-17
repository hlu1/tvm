from nnvm.frontend import from_caffe2
from caffe2.proto import caffe2_pb2
from caffe2.proto import metanet_pb2
from caffe2.python.predictor_constants import predictor_constants
from c2_models import load_caffe2_metanet, models
import click

@click.command()
@click.option('--model', type=click.Choice(['seg', 'seg_le']), required=True)

def convert_metanetdef_to_netdef(model):
    init_net, predict_net = load_caffe2_metanet(models[model].init_net, models[model].predict_net)
    for index in range(len(init_net.nets)):
        try:
            init = init_net.nets[index].value
            predict = predict_net.nets[index].value
            with open(models[model].init_net + '.' + str(index) + '.pb', 'wb') as f:
                f.write(init.SerializeToString())
            with open(models[model].predict_net + '.' + str(index) + '.pb', 'wb') as f:
                f.write(predict.SerializeToString())
        except:
            print('Oops, something went wrong')

if __name__ == '__main__':
    convert_metanetdef_to_netdef()
