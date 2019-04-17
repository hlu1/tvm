import numpy as np
import tvm
from tvm import autotvm, relay
from tvm.autotvm.task.space import FallbackConfigEntity
from tvm.contrib import graph_runtime
import nnvm
import mxnet as mx


def unet(alignment=None):
    def conv(x, num_filter):
        return mx.sym.Convolution(x, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_filter=num_filter)

    def pool(x):
        return mx.sym.Pooling(x, kernel=(2, 2), stride=(2,2), pad=(0, 0), pool_type='max')

    def resize(x):
        return mx.sym.UpSampling(x, scale=2, sample_type="nearest")

    def relu(x):
        return mx.sym.relu(x)

    def sigmoid(x):
        return mx.sym.sigmoid(x)

    def conv_pool(x, f):
        c = conv(x, f)
        p = pool(c)
        r = relu(p)
        return (r, c)

    def conv_relu(x, f):
        c = conv(x, f)
        r = relu(c)
        return r

    def conv_resize(x, f):
        c = conv(x, f)
        r = resize(c)
        return r

    def align(x):
        return x if not alignment else ((x + alignment - 1) // alignment) * alignment

    data = mx.sym.Variable(name='data')
    (r0, p0) = conv_pool(data, align(12))  # 96x96
    (r1, p1) = conv_pool(r0, align(24))  # 48x48
    (r2, p2) = conv_pool(r1, align(48))  # 24x24
    (r3, p3) = conv_pool(r2, align(96))  # 12x12
    (r4, p4) = conv_pool(r3, align(180))  # 6x6
    r5 = conv_relu(r4, align(220))  # 6x6
    r6 = relu(conv_resize(r5, align(180)) + p4)
    r7 = relu(conv_resize(r6, align(96)) + p3)
    r8 = relu(conv_resize(r7, align(48)) + p2)
    r9 = relu(conv_resize(r8, align(24)) + p1)
    r10 = relu(conv_resize(r9, align(12)) + p0)
    r11 = conv(r10, 1)
    s = sigmoid(r11)
    return s

def get_mxnet_output(symbol, x, dtype='float32'):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    mod = mx.mod.Module(symbol, context=mx.cpu(), label_names=None)
    mod.bind(data_shapes=[('data', x.shape)], for_training=False)
    mod.init_params()
    mod.forward(Batch([mx.nd.array(x.astype(dtype))]))
    out = mod.get_outputs()[0].asnumpy()
    args, auxs = mod.get_params()
    return out, args, auxs

class WinogradFallback(autotvm.FallbackContext):
    def _query_inside(self, target, workload):
        key = (target, workload)
        if key in self.memory:
            return self.memory[key]
        cfg = FallbackConfigEntity()
        cfg.template_key = 'winograd_nnpack_fp32'
        self.memory[key] = cfg
        return cfg

def get_nnvm_output(symbol, x, args, auxs, target, ctx, out_shape, dtype='float32'):
    new_sym, params = nnvm.frontend.from_mxnet(symbol, args, auxs)
    dshape = x.shape
    shape_dict = {'data': dshape}
    with WinogradFallback():
        with tvm.target.create(target):
            with nnvm.compiler.build_config(opt_level=3):
                graph, lib, params = nnvm.compiler.build(new_sym, target, shape_dict, params=params)
            m = graph_runtime.create(graph, lib, ctx)
            # set inputs
            m.set_input("data", tvm.nd.array(x.astype(dtype)))
            m.set_input(**params)
            m.run()
            # get outputs
            out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
            return out.asnumpy()

def get_relay_output(symbol, x, args, auxs, target, ctx, out_shape, dtype='float32'):
    shape_dict = {"data": x.shape}
    new_sym, params = relay.frontend.from_mxnet(symbol,
                                                shape_dict,
                                                arg_params=args,
                                                aux_params=auxs)
    with WinogradFallback():
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(new_sym, target, params=params)
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input("data", tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()
    # get outputs
    out = m.get_output(0, tvm.nd.empty(out_shape, dtype))
    return out.asnumpy()

def verify_unet(data_shape=(1, 3, 192, 192), out_shape=(1, 1, 192, 192)):
    mx_sym = unet()
    x = np.random.uniform(size=data_shape)
    dtype = 'float32'
    mx_out, args, auxs = get_mxnet_output(mx_sym, x, dtype)
    assert "data" not in args
    target = 'llvm -device=arm_cpu'
    ctx = tvm.context(target, 0)

    relay_out = get_relay_output(mx_sym, x, args, auxs, target, ctx, out_shape, dtype)
    tvm.testing.assert_allclose(mx_out, relay_out, rtol=1e-5, atol=1e-5)

    nnvm_out = get_nnvm_output(mx_sym, x, args, auxs, target, ctx, out_shape, dtype)
    tvm.testing.assert_allclose(mx_out, nnvm_out, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    autotvm.DispatchContext.current.silent = True

    verify_unet(data_shape=(1, 3, 192, 192))
    verify_unet(data_shape=(1, 4, 192, 192))
    verify_unet(data_shape=(1, 3, 96, 96), out_shape=(1, 1, 96, 96))
    verify_unet(data_shape=(1, 4, 96, 96), out_shape=(1, 1, 96, 96))
