"""x86 dense operators"""
from __future__ import absolute_import as _abs
import tvm

from .util import get_fp32_len
from .. import generic, tag, nn
from ..util import traverse_inline, get_const_tuple
from tvm.contrib import cblas

import numpy as np
from tvm import relay


@nn.dense.register("cpu")
def dense(data, weight, bias=None, out_dtype=None, transposed=False):
    matmul = cblas.matmul(data, weight, transb=not transposed, tag="dense_blas")
    if bias is not None:
        matmul = tvm.compute(
            get_const_tuple(matmul.shape),
            lambda i, j: matmul[i, j] + bias[j],
            name="bias_add",
            tag=tag.BROADCAST
        )
    return matmul


@generic.schedule_dense.register("cpu")
def schedule_dense(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'dense_blas':
            data, weight = op.input_tensors
            # vectorize fused op
            if outs[0].op != op:
                (_, y) = s[outs[0].op].op.axis
                s[outs[0].op].vectorize(y)

    traverse_inline(s, outs[0].op, _callback)

    return s


@nn.dense_alter_layout.register("cpu")
def dense_alter_layout(attrs, inputs, tinfos):
    assert len(inputs) == 2
    data, weight = inputs
    new_attrs = dict(attrs)
    if not new_attrs["transposed"]:
        new_attrs["transposed"] = True
        transposed_weight = relay.transpose(
            weight,
            axes=(1, 0),
        )
        return relay.nn.dense(data, transposed_weight, **new_attrs)
    else:
        new_attrs["transposed"] = True
        return relay.nn.dense(data, weight, **new_attrs)
