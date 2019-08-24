# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import tvm
from tvm.contrib import graph_runtime
from tvm.relay.testing.config import ctx_list
from tvm import relay
from model_zoo import c2_squeezenet, c2_resnet50, c2_vgg19
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2
from collections import namedtuple


def get_tvm_output(model,
                   input_data,
                   target,
                   ctx,
                   output_shape,
                   output_dtype='float32'):
    """ Generic function to execute and get tvm output"""
    # supporting multiple inputs in caffe2 in a bit tricky,
    # because the input names can appear at the beginning or end of model.predict_net.external_input
    assert isinstance(input_data, np.ndarray)

    # here we use the first input blob to the first op to get the input name
    input_names = model.predict_net.op[0].input[0]
    shape_dict = {input_names: input_data.shape}
    dtype_dict = {input_names: input_data.dtype}
    mod, params = relay.frontend.from_caffe2(
        model.init_net, model.predict_net, shape_dict, dtype_dict)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    m = graph_runtime.create(graph, lib, ctx)

    # set inputs
    m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input(**params)

    # execute
    m.run()

    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0, tvm.nd.empty((output_shape),
                                                  output_dtype))
        return tvm_output.asnumpy()


def get_caffe2_output(model, x, dtype='float32'):
    workspace.RunNetOnce(model.init_net)

    input_blob = model.predict_net.op[0].input[0]
    workspace.FeedBlob(input_blob, x.astype(dtype))
    workspace.RunNetOnce(model.predict_net)

    output_blob = model.predict_net.external_output[0]
    c2_output = workspace.FetchBlob(output_blob)
    return c2_output


def verify_caffe2_forward_impl(model, data_shape):
    dtype = 'float32'
    data = np.random.uniform(size=data_shape).astype(dtype)
    c2_out = get_caffe2_output(model, data, dtype)
    out_shape = c2_out.shape
    print(out_shape)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, data, target, ctx, out_shape, dtype)
        tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)


def test_forward_squeezenet1_1():
    verify_caffe2_forward_impl(c2_squeezenet, (1, 3, 224, 224))


def test_forward_resnet50():
    verify_caffe2_forward_impl(c2_resnet50, (1, 3, 224, 224))


def test_forward_vgg19():
    verify_caffe2_forward_impl(c2_vgg19, (1, 3, 224, 224))


Model = namedtuple('Model', ['init_net', 'predict_net'])


def test_elementwise_op(op):
    assert isinstance(op, str)
    data_shape = (1, 16, 9, 9)
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['A', 'B']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['A'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['B'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ['A', 'B']
    predict_net.external_output[:] = ['C']
    predict_net.op.extend([
        core.CreateOperator(
            op,
            ['A', 'B'],
            ['C'],
        )
    ])

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)


def test_elementwise_add_with_broadcast():
    data_shape = (1, 16, 9, 9)
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['A', 'B']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['A'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['B'],
            shape=(1,),
            values=np.random.uniform(size=1).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ['A', 'B']
    predict_net.external_output[:] = ['C']
    predict_net.op.extend([
        core.CreateOperator(
            'Add',
            ['A', 'B'],
            ['C'],
            broadcast=1,
        )
    ])

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)


def test_normalize_yuv():
    data_shape = (1, 3, 96, 96)
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['A', 'mean', 'std']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['A'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['mean'],
            shape=(1, 3,),
            values=np.random.uniform(size=3).flatten().tolist(),
        ),
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['std'],
            shape=(1, 3,),
            values=np.random.uniform(size=3).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ['A', 'mean', 'std']
    predict_net.external_output[:] = ['C']
    predict_net.op.extend([
        core.CreateOperator(
            'NormalizePlanarYUV',
            ['A', 'mean', 'std'],
            ['C'],
        )
    ])

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)


def test_dense(pretransposed=False):
    M, N, K = 13, 7, 16
    for data_shape in ((M, K), (M, K, 1, 1)):
        weight_shape = (K, N) if pretransposed else (N, K)
        bias_shape = (N,)
        init_net = caffe2_pb2.NetDef()
        init_net.name = 'test_init_net'
        init_net.external_output[:] = ['I', 'W', 'B']
        init_net.op.extend([
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['I'],
                shape=data_shape,
                values=np.random.uniform(size=data_shape).flatten().tolist(),
            ),
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['W'],
                shape=weight_shape,
                values=np.random.uniform(size=weight_shape).flatten().tolist(),
            ),
            core.CreateOperator(
                'GivenTensorFill',
                [],
                ['B'],
                shape=bias_shape,
                values=np.random.uniform(size=bias_shape).flatten().tolist(),
            ),
        ])

        predict_net = caffe2_pb2.NetDef()
        predict_net.name = 'test_predict_net'
        predict_net.external_input[:] = ['I', 'W', 'B']
        predict_net.external_output[:] = ['Y']
        predict_net.op.extend([
            core.CreateOperator(
                'FCTransposed' if pretransposed else 'FC',
                ['I', 'W', 'B'],
                ['Y'],
            )
        ])

        model = Model(init_net, predict_net)
        verify_caffe2_forward_impl(model, data_shape)


def test_batch_matmul(trans_a=0, trans_b=1):
    B, M, N, K = 4, 13, 7, 16
    data_shape = (B, M, K) if not trans_a else (B, K, M)
    weight_shape = (B, N, K) if trans_b else (B, K, N)
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['I', 'W']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['I'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['W'],
            shape=weight_shape,
            values=np.random.uniform(size=weight_shape).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ['I', 'W']
    predict_net.external_output[:] = ['Y']
    predict_net.op.extend([
        core.CreateOperator(
            'BatchMatMul',
            ['I', 'W'],
            ['Y'],
            trans_a=trans_a,
            trans_b=trans_b,
        )
    ])

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)


def test_reshape(m, n, shape):
    data_shape = (m, n)
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['X']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['X'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ["X"]
    predict_net.external_output[:] = ["Y"]
    predict_net.op.add().CopyFrom(
        core.CreateOperator(
            "Reshape",
            ["X"],
            ["Y", "old_shape"],
            shape=shape,
        )
    )

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)


def test_flatten(data_shape, axis=1):
    init_net = caffe2_pb2.NetDef()
    init_net.name = 'test_init_net'
    init_net.external_output[:] = ['X']
    init_net.op.extend([
        core.CreateOperator(
            'GivenTensorFill',
            [],
            ['X'],
            shape=data_shape,
            values=np.random.uniform(size=data_shape).flatten().tolist(),
        ),
    ])

    predict_net = caffe2_pb2.NetDef()
    predict_net.name = 'test_predict_net'
    predict_net.external_input[:] = ["X"]
    predict_net.external_output[:] = ["Y"]
    predict_net.op.add().CopyFrom(
        core.CreateOperator(
            "Flatten",
            ["X"],
            ["Y"],
            axis=axis,
        )
    )

    model = Model(init_net, predict_net)
    verify_caffe2_forward_impl(model, data_shape)



if __name__ == '__main__':
    test_forward_squeezenet1_1()
    test_forward_resnet50()
    test_forward_vgg19()

    test_elementwise_op('Add')
    test_elementwise_op('Mul')
    test_elementwise_add_with_broadcast()
    test_normalize_yuv()
    test_dense()
    test_dense(pretransposed=True)
    test_batch_matmul(0, 0)
    test_batch_matmul(0, 1)
    test_batch_matmul(1, 0)
    test_batch_matmul(1, 1)
    test_reshape(7, 16, [7, -1, 1])
    test_reshape(7, 16, [7, -1, 4, 1])
    test_flatten([4, 5])
    test_flatten([4, 5], 0)
    test_flatten([4, 5, 6, 8], 0)
    test_flatten([4, 5, 6, 8], 1)
    test_flatten([4, 5, 6, 8], 2)
    test_flatten([4, 5, 6, 8], 3)
