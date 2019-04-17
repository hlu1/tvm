import numpy as np
import nnvm
import tvm
from tvm.contrib import graph_runtime
from nnvm.testing.config import ctx_list

from caffe2.python import workspace
from c2_models import get_c2_model, get_caffe2_output


def get_tvm_output(model,
                   input_data,
                   target,
                   ctx,
                   output_shape,
                   output_dtype='float32'):
    """ Generic function to execute and get tvm output"""
    sym, params = nnvm.frontend.from_caffe2(model.init_net, model.predict_net)

    # supporting multiple inputs in caffe2 in a bit tricky,
    # because the input names can appear at the beginning or end of model.predict_net.external_input
    assert isinstance(input_data, np.ndarray)

    # here we use the first input blob to the first op to get the input name
    input_names = model.predict_net.op[0].input[0]
    shape_dict = {input_names: input_data.shape}
    dtype_dict = {input_names: input_data.dtype}

    with nnvm.compiler.build_config(opt_level=2):
        with tvm.build_config(partition_const_loop=True):
            graph, lib, params = nnvm.compiler.build(
                sym, target, shape=shape_dict, dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    m = graph_runtime.create(graph, lib, ctx)

    # set inputs
    m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input(**params)

    # execute
    m.run()

    ftimer = m.module.time_evaluator("run", ctx, 5)
    for i in range(1):
        prof_res = ftimer()
        # time.sleep(1)

    for i in range(2):
        prof_res = ftimer()
        print("TVM time: ", prof_res.mean)

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


def verify_caffe2_forward_impl(model, data_shape, out_shape):
    dtype = 'float32'
    data = np.random.uniform(size=data_shape).astype(dtype)
    c2_out = get_caffe2_output(model, data, dtype)
    for target, ctx in ctx_list():
        tvm_out = get_tvm_output(model, data, target, ctx, out_shape, dtype)
        tvm.testing.assert_allclose(c2_out, tvm_out, rtol=1e-5, atol=1e-5)


def verify_unet(m):
    model = get_c2_model(m)
    verify_caffe2_forward_impl(model, tuple([1] + list(model.image_shape)), tuple([1] + list(model.output_shape)))


if __name__ == '__main__':
    verify_unet("seg_detect")
    # verify_unet("seg_track")
