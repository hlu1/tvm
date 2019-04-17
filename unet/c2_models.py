from nnvm.frontend import from_caffe2
from tvm import relay
from caffe2.proto import caffe2_pb2
from caffe2.proto import metanet_pb2
from caffe2.python.predictor_constants import predictor_constants
from caffe2.python import workspace
from collections import namedtuple


Model = namedtuple("Model", ["init_net", "predict_net", "image_shape", "output_shape", "is_meta_net"])
models = {
    "seg_detect": Model(
        "model/105/seg_detect_init_net.pb",
        "model/105/seg_detect_predict_net.pb",
        (3, 192, 192),
        (1, 192, 192),
        False,
    ),
    "seg_track": Model(
        "model/105/seg_track_init_net.pb",
        "model/105/seg_track_predict_net.pb",
        (4, 192, 192),
        (1, 192, 192),
        False,
    ),
    "seg": Model(
        "model/106/he/seg_init_net.pb",
        "model/106/he/seg_predict_net.pb",
        [(3, 192, 192), (4, 192, 192),],
        [(1, 192, 192), (1, 192, 192),],
        True,
    ),
    "seg_le": Model(
        "model/106/le/seg_init_net.pb",
        "model/106/le/seg_predict_net.pb",
        [(3, 96, 96), (4, 96, 96),],
        [(1, 96, 96), (1, 96, 96),],
        True,
    ),
    "seg_le_detect": Model(
        "model/106/le/seg_le_detect_init_net.pb",
        "model/106/le/seg_le_detect_predict_net.pb",
        (3, 96, 96),
        (1, 96, 96),
        False,
    ),
    "seg_le_track": Model(
        "model/106/le/seg_le_track_init_net.pb",
        "model/106/le/seg_le_track_predict_net.pb",
        (4, 96, 96),
        (1, 96, 96),
        False,
    ),
}


def load_caffe2_netdef(init, predict):
    init_net = caffe2_pb2.NetDef()
    with open(init, "rb") as f:
        init_net.ParseFromString(f.read())

    predict_net = caffe2_pb2.NetDef()
    with open(predict, "rb") as f:
        predict_net.ParseFromString(f.read())
    return init_net, predict_net


def load_caffe2_metanet(init, predict):
    init_net = metanet_pb2.MetaNetDef()
    with open(init, 'rb') as f:
        init_net.ParseFromString(f.read())

    predict_net = metanet_pb2.MetaNetDef()
    with open(predict, 'rb') as f:
        predict_net.ParseFromString(f.read())

    return init_net, predict_net


def get_c2_model(model):
    init_net, predict_net = load_caffe2_netdef(
        models[model].init_net, models[model].predict_net
    )
    return Model(
        init_net, predict_net, models[model].image_shape, models[model].output_shape, False
    )


def get_nnvm_sym(model):
    init_net, predict_net = load_caffe2_netdef(
        models[model].init_net, models[model].predict_net
    )
    sym, params = from_caffe2(init_net, predict_net)
    return sym, params, models[model].image_shape, models[model].output_shape


def get_relay_func(model, dtype="float32"):
    init_net, predict_net = load_caffe2_netdef(
        models[model].init_net, models[model].predict_net
    )
    input_name = predict_net.op[0].input[0]
    shape_dict = {input_name: tuple([1] + list(models[model].image_shape))}
    dtype_dict = {input_name: dtype}
    func, params = relay.frontend.from_caffe2(init_net, predict_net, shape_dict, dtype_dict)
    return func, params, shape_dict, models[model].output_shape


def get_caffe2_output(model, x, dtype="float32"):
    workspace.RunNetOnce(model.init_net)

    input_blob = model.predict_net.op[0].input[0]
    workspace.FeedBlob(input_blob, x.astype(dtype))
    workspace.RunNetOnce(model.predict_net)

    output_blob = model.predict_net.external_output[0]
    c2_output = workspace.FetchBlob(output_blob)
    return c2_output
