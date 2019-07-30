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
# pylint: disable=invalid-name,too-many-locals,unused-variable
"""x86 dense operators"""
from __future__ import absolute_import as _abs
import logging

import tvm
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas

from .util import get_fp32_len
from .. import generic, tag, nn
from ..nn import dense, dense_alter_layout
from ..util import traverse_inline, get_const_tuple

[DIRECT, DIRECT_PACK, BLAS, BLAS_PRETRANSPOSED] = ALGORITHMS = range(4)

@autotvm.register_topi_compute(nn.dense, "cpu", "direct")
def _declaration_dense(cfg, data, weight, bias=None, kernel_layout="OI", out_dtype=None):
    target = tvm.target.current_target()
    # use AutoTVM to pick the best algorithm
    if 'cblas' in target.libs:
        cfg.define_knob('algo', ALGORITHMS)
    else:
        cfg.define_knob('algo', ALGORITHMS[:2])
    f = [_declaration_dense_nopack, _declaration_dense_pack, _declaration_dense_blas,
            _declaration_dense_blas_pretranspose][cfg['algo'].val]
    return f(cfg, data, weight, bias, kernel_layout, out_dtype)


# Declare dense compute with packing weight into cache-friendly layout
@autotvm.register_topi_compute(nn.dense, "cpu", "direct_pack")
def _declaration_dense_pack(cfg, data, weight, bias=None, kernel_layout="OI", out_dtype=None):
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=3)
    cfg.define_split("tile_x", out_dim, num_outputs=3)
    cfg.define_split("tile_k", in_dim, num_outputs=2)
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, batch, out_dim, in_dim)

    packw_bn = cfg["tile_x"].size[-1]
    packw_shape = (out_dim // packw_bn, in_dim, packw_bn)
    packw = tvm.compute(packw_shape,
                        lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight")

    k = tvm.reduce_axis((0, in_dim), name="k")
    C = tvm.compute((batch, out_dim),
                    lambda y, x: tvm.sum(
                        data[y, k].astype(out_dtype) *
                        packw[x // packw_bn, k, x % packw_bn].astype(out_dtype),
                        axis=k),
                    tag="dense_pack")
    if bias is not None:
        C = tvm.compute((batch, out_dim), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)
    return C


# Declare dense compute without packing weight
@autotvm.register_topi_compute(nn.dense, "cpu", "direct_nopack")
def _declaration_dense_nopack(cfg, data, weight, bias=None, kernel_layout="OI", out_dtype=None):
    assert kernel_layout == "OI"
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split("tile_x", out_dim, num_outputs=2)
    cfg.define_split("tile_y", batch, num_outputs=2)
    cfg.define_split("tile_k", in_dim, num_outputs=2)
    if cfg.is_fallback:
        _default_dense_nopack_config(cfg, batch, out_dim, in_dim)

    vec = cfg["tile_k"].size[-1]
    k = tvm.reduce_axis((0, in_dim // vec), "k")
    CC = tvm.compute((batch, out_dim, vec),
                        lambda z, y, x: tvm.sum(
                            data[z, k * vec + x].astype(out_dtype) *
                            weight[y, k * vec + x].astype(out_dtype), axis=k))

    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((batch, out_dim),
                    lambda y, x: tvm.sum(CC[y, x, kk], axis=kk),
                    tag="dense_nopack")
    if bias is not None:
        C = tvm.compute((batch, out_dim), lambda i, j: C[i, j] + bias[j].astype(out_dtype),
                        tag=tag.BROADCAST)

    return C


@autotvm.register_topi_schedule(generic.schedule_dense, "cpu", ["direct", "direct_pack",
    "direct_nopack"])
def _schedule_dense(cfg, outs):
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _schedule_dense_pack_template(cfg, s, op.output(0))
        elif 'dense_nopack' in op.tag:
            _schedule_dense_nopack_template(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_pack_template(cfg, s, C):
    A, packedB = s[C].op.input_tensors

    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis

    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packedB].op.axis
    s[packedB].reorder(z, x, y)
    s[packedB].parallel(z)
    s[packedB].vectorize(y)
    return s


def _schedule_dense_nopack_template(cfg, s, C):
    y, x = s[C].op.axis
    kk, = s[C].op.reduce_axis
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    CC, = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    return s


def _default_dense_pack_config(cfg, M, N, K):
    vec_width = get_fp32_len()

    tilex_ii = 1
    for bn in range(vec_width*2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K, 1])


def _default_dense_nopack_config(cfg, M, N, K):
    vec_width = get_fp32_len()
    tilek_bn = 1
    for bn in range(vec_width*2, 0, -1):
        if K % bn == 0:
            tilek_bn = bn
            break
    cfg["tile_k"] = SplitEntity([K // tilek_bn, tilek_bn])
    cfg["tile_x"] = SplitEntity([N, 1])
    cfg["tile_y"] = SplitEntity([1, M])


@autotvm.register_topi_compute(nn.dense, "cpu", "blas")
def _declaration_dense_blas(cfg, data, weight, bias=None, kernel_layout="OI", out_dtype=None):
    matmul = cblas.matmul(data, weight, transb=True, tag="dense_blas")
    if bias is not None:
        matmul = tvm.compute(
            get_const_tuple(matmul.shape),
            lambda i, j: matmul[i, j] + bias[j],
            name="bias_add",
            tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_compute(nn.dense, "cpu", "blas_pretransposed")
def _declaration_dense_blas_pretranspose(cfg, data, weight, bias=None, kernel_layout="OI", out_dtype=None):
    import topi
    if kernel_layout == "OI":
        weight = topi.transpose(weight, [1, 0])
    else:
        assert kernel_layout == "IO"
    matmul = cblas.matmul(data, weight, transb=False, tag="blas_pretransposed")
    if bias is not None:
        matmul = tvm.compute(
            get_const_tuple(matmul.shape),
            lambda i, j: matmul[i, j] + bias[j],
            name="bias_add",
            tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_schedule(generic.schedule_dense, 'cpu', ['blas', 'blas_pretransposed'])
def _schedule_dense(cfg, outs):
    """Schedule for dense operator.
    Parameters
    ----------
    cfg: ConfigEntity
        The config entity for this template
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.
     Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'blas_pretransposed':
            data, weight = op.input_tensors
            if autotvm.GLOBAL_SCOPE.in_tuning:
                s[weight].pragma(s[weight].op.axis[0], "debug_skip_region")
        if outs[0].op != op:
            s[outs[0].op].vectorize(s[outs[0]].op.axis[-1])

    traverse_inline(s, outs[0].op, _callback)
    return s


@nn.dense_alter_layout.register("cpu")
def dense_alter_layout(attrs, inputs, tinfos, F):
    copy_inputs = [i for i in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}

    data, weight = tinfos[:2]
    bias = tinfos[2] if len(tinfos) == 3 else None
    kernel_layout = attrs.get_str("kernel_layout")
    out_dtype = attrs["out_dtype"]
    if out_dtype in ("same", ""):
        out_dtype = tinfos[0].dtype

    # query config of this workload
    workload = autotvm.task.args_to_workload(
        [data, weight, bias, kernel_layout, out_dtype], dense)
    target = tvm.target.current_target()
    dispatch_ctx = autotvm.DispatchContext.current
    cfg = dispatch_ctx.query(target, workload)

    if cfg.is_fallback:
        return None

    if cfg.template_key == "blas_pretransposed":
        new_attrs["kernel_layout"] = "IO"
        N, C = get_const_tuple(weight.shape)
        new_weight = tvm.placeholder((C, N), dtype=weight.dtype)
        new_workload = autotvm.task.args_to_workload(
            [data, new_weight, bias, new_attrs["kernel_layout"], out_dtype], dense)
        dispatch_ctx.update(target, new_workload, cfg)
    else:
        # do not apply alter_op_layout
        return None

    if F.__name__ == "nnvm.symbol":
        logging.warning("Use native layout for dense on NNVM.")
        return None

    return F.nn.dense(*copy_inputs, **new_attrs)
