import numpy as np
import tvm
import topi
import logging
from tvm import relay
import dense

logging.basicConfig(level=logging.DEBUG)

def test_relu(m, n):
    A = tvm.placeholder((m, n), name='A')
    B = topi.nn.relu(A)
    device = "llvm"
    with tvm.target.create(device):
        s = topi.generic.schedule_elemwise(B)

    f = tvm.build(s, [A, B], device, name="relu")
    print(tvm.lower(s, [A, B], simple_mode=True))

def test_dense_add_relu(m, k, n):
    a = relay.var("a", shape=(m, k))
    b = relay.var("b", shape=(n, k))
    c = relay.var("c", shape=(n,))
    out = relay.nn.relu(relay.nn.bias_add(relay.nn.dense(a, b), c))
    f = relay.ir_pass.infer_type(relay.Function([a, b, c], out))
    opt_level = 3
    target = "llvm"
    with relay.build_config(opt_level=opt_level):
        graph, lib, params = relay.build(f, target, params={})
    print(graph)

# test_relu(1024, 128)

test_dense_add_relu(16, 24, 8)
