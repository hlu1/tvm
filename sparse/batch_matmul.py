import tvm
from tvm import autotvm
import topi
from topi import generic, nn, tag
from topi.util import traverse_inline, get_const_tuple, get_max_power2_factor

from tvm.contrib import cblas

@autotvm.register_topi_compute(nn.batch_matmul, 'cpu', ['direct'])
def batch_matmul(cfg, A, B):
    assert len(A.shape) == len(B.shape), \
        "Shape mismatch between inputs"
    A_shape = get_const_tuple(A.shape)
    B_shape = get_const_tuple(B.shape)
    Batch, M, K = A_shape
    N = B_shape[-2]
    oshape = (Batch, M, N)

    cfg.define_knob('blas', [0, 1, 2])
    f = [batch_matmul_direct, batch_matmul_blas, batch_matmul_blas][cfg['blas'].val]
    C = f(cfg, A, B)
    cfg.add_flop(2 * oshape[0] * oshape[1] * oshape[2] * K)
    return C


def batch_matmul_direct(cfg, A, B):
    """The default implementation of batched_matmul in topi.

    Parameters
    ----------
    A: tvm.Tensor
        n-D with shape [B, M, K]

    B: tvm.Tensor
        n-D with shape [B, N, K]

    Returns
    -------
    output: tvm.Tensor
        n-D with shape [B, M, N]
    """
    A_shape = get_const_tuple(A.shape)
    B_shape = get_const_tuple(B.shape)
    Batch, M, K = A_shape
    N = B_shape[-2]
    oshape = (Batch, M, N)
    k = tvm.reduce_axis((0, K), name='k')
    return tvm.compute(oshape,
                       lambda b, x, y: tvm.sum(A[b, x, k] * B[b, y, k], axis=k),
                       tag='batch_matmul')

def batch_matmul_blas(cfg, A, B):
    iterative = cfg['blas'].val == 2
    return cblas.batch_matmul(
        A, B, transa=False, transb=True, iterative=iterative, tag="batch_matmul")

@autotvm.register_topi_schedule(generic.schedule_batch_matmul, 'cpu', ['direct'], override=True)
def schedule_batch_matmul(cfg, outs):
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
        if "batch_matmul" in op.tag:
            if outs[0].op != op:
                (_, x, y) = s[outs[0].op].op.axis
                s[outs[0].op].vectorize(y)

    traverse_inline(s, outs[0].op, _callback)
    return s
