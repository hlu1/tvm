from __future__ import absolute_import, print_function

import click
import numpy as np
import topi
import tvm
from tvm import te


TARGETS = dict(
    skl="llvm -mcpu=skylake -target=x86_64-linux-gnu",
    skl_512="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu",
    brw="llvm -mcpu=broadwell -target=x86_64-linux-gnu",
)


def deep_wide(target, batch_size=128, split_concat=True, num_features=50, emb_size=32):
    ad_emb = te.placeholder((batch_size, emb_size), name="ad_emb")
    usr_emb = te.placeholder((1, emb_size), name="usr_emb")
    wide = te.placeholder((batch_size, num_features), name="wide")

    fc_w = te.placeholder((1, num_features + 1), name="fc_w")
    fc_w_dp = te.placeholder((1, 1), name="fc_w_dp")
    fc_w_wide = te.placeholder((1, num_features), name="fc_w_wide")
    fc_b = te.placeholder((1,), name="fc_b")

    # Add + Mul, ReplaceNaN, Clip
    mu = te.placeholder((1, num_features), name="mu")
    sigma = te.placeholder((1, num_features), name="sigma")

    wide_normalized = topi.multiply(topi.add(wide, mu), sigma)
    wide_noNaN = te.compute(
        (batch_size, num_features),
        lambda i, j: tvm.tir.if_then_else(
            topi.isnan(wide_normalized)[i, j], 0, wide_normalized[i, j]
        ),
        name="wide_noNaN",
    )
    # wide_noNaN = topi.where(topi.isnan(wide_normalized), topi.full((batch_size, num_features), wide_normalized.dtype, 0), wide_normalized)
    # wide_noNaN = wide_normalized
    wide_preproc = topi.clip(wide_noNaN, -10.0, 10.0)

    # batch_matmul + flatten => matmul
    k1 = te.reduce_axis((0, emb_size), "k1")
    dp = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(ad_emb[i, k1] * usr_emb[j, k1], axis=k1),
        name="dp",
    )

    # concat, fc => fc_dp, fc_wide, add

    if split_concat:
        # fc_dp
        fc_dp = te.compute(
            (batch_size, 1), lambda i, j: dp[i, j] * fc_w_dp[0, 0], name="fc_dp"
        )
        # fc_wide
        k2 = te.reduce_axis((0, num_features), "k2")
        fc = te.compute(
            (batch_size, 1),
            lambda i, j: te.sum(wide_preproc[i, k2] * fc_w_wide[j, k2], axis=k2),
            name="fc_wide",
        )
        # add
        fc_out = topi.add(topi.add(fc_dp, fc), fc_b)
    else:
        # concat, fc
        concat = topi.concatenate((dp, wide_preproc), axis=1)
        k2 = te.reduce_axis((0, num_features + 1), "k2")
        fc = te.compute(
            (batch_size, 1), lambda i, j: te.sum(concat[i, k2] * fc_w[j, k2], axis=k2)
        )
        fc_out = topi.add(fc, fc_b)

    # sigmoid => relu
    out = topi.sigmoid(fc_out)
    # out = topi.nn.relu(fc_out)

    s = te.create_schedule(out.op)
    te.schedule.AutoInlineInjective(s)
    x, y = s[out].op.axis
    fused = s[out].fuse(x, y)

    if batch_size > 1:
        # m is batch_size
        (m, n) = s[fc].op.axis
        (k,) = s[fc].op.reduce_axis
        # TUNABLE
        (mo, mi) = s[fc].split(m, factor=16)
        s[fc].reorder(mo, n, k, mi)
        (m, n) = s[dp].op.axis
        (k,) = s[dp].op.reduce_axis
        # TUNABLE
        (mo, mi) = s[dp].split(m, factor=16)
        s[dp].reorder(mo, n, k, mi)

    if split_concat:
        args_tvm = [ad_emb, usr_emb, wide, mu, sigma, fc_w_dp, fc_w_wide, fc_b, out]
    else:
        args_tvm = [ad_emb, usr_emb, wide, mu, sigma, fc_w, fc_b, out]

    func = tvm.build(s, args_tvm, target=target, name="fused_op")
    print(tvm.lower(s, args_tvm, simple_mode=True))
    # print(func.get_source('asm'))

    ad_emb_np = np.random.uniform(size=(batch_size, emb_size)).astype(ad_emb.dtype)
    usr_emb_np = np.random.uniform(size=(1, emb_size)).astype(usr_emb.dtype)
    wide_np = np.random.uniform(size=(batch_size, num_features)).astype(wide.dtype)
    mu_np = np.random.uniform(size=(1, num_features)).astype(mu.dtype)
    sigma_np = np.random.uniform(size=(1, num_features)).astype(sigma.dtype)
    fc_w_np = np.random.uniform(size=(1, num_features + 1)).astype(fc_w.dtype)
    fc_w_dp_np = np.random.uniform(size=(1, 1)).astype(fc_w_dp.dtype)
    fc_w_wide_np = np.random.uniform(size=(1, num_features)).astype(fc_w_wide.dtype)
    fc_b_np = np.random.uniform(size=(1,)).astype(fc_b.dtype)
    out_np = np.zeros((batch_size, 1)).astype("float32")

    ctx = tvm.context(target, 0)
    ad_emb_nd = tvm.nd.array(ad_emb_np, ctx)
    usr_emb_nd = tvm.nd.array(usr_emb_np, ctx)
    wide_nd = tvm.nd.array(wide_np, ctx)
    mu_nd = tvm.nd.array(mu_np, ctx)
    sigma_nd = tvm.nd.array(sigma_np, ctx)
    fc_w_nd = tvm.nd.array(fc_w_np, ctx)
    fc_w_dp_nd = tvm.nd.array(fc_w_dp_np, ctx)
    fc_w_wide_nd = tvm.nd.array(fc_w_wide_np, ctx)
    fc_b_nd = tvm.nd.array(fc_b_np, ctx)
    out_nd = tvm.nd.array(out_np, ctx)

    if split_concat:
        eval_args = (
            ad_emb_nd,
            usr_emb_nd,
            wide_nd,
            mu_nd,
            sigma_nd,
            fc_w_dp_nd,
            fc_w_wide_nd,
            fc_b_nd,
            out_nd,
        )
    else:
        eval_args = (
            ad_emb_nd,
            usr_emb_nd,
            wide_nd,
            mu_nd,
            sigma_nd,
            fc_w_nd,
            fc_b_nd,
            out_nd,
        )

    func(*eval_args)
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3, min_repeat_ms=100)
    mean = evaluator(*eval_args).mean
    flops = tvm.autotvm.task.task.compute_flop(s)
    print(
        "batch_size: ",
        batch_size,
        "time: ",
        mean * 1e6,
        ", CPU per ad: ",
        mean / batch_size * 1e6,
        "us, GFlops: ",
        flops / mean / 1e9,
    )


def deep_wide_transpose(target, batch_size=128, num_features=100, emb_size=32):
    ad_emb = te.placeholder((emb_size, batch_size), name="ad_emb")
    usr_emb = te.placeholder((emb_size, 1), name="usr_emb")
    wide = te.placeholder((num_features, batch_size), name="wide")

    fc_w_emb = te.placeholder((1, num_features), name="fc_w_emb")
    fc_w_dp = te.placeholder((1, 1), name="fc_w_dp")
    fc_b = te.placeholder((1,), name="fc_b")

    k1 = te.reduce_axis((0, emb_size), "k1")
    dp = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(ad_emb[k1, i] * usr_emb[k1, j], axis=k1),
        name="dp",
    )
    concat = topi.concatenate((dp, wide), axis=1)
    k2 = te.reduce_axis((0, num_features), "k2")
    k_dp = te.reduce_axis((0, 1), "k_dp")
    fc_dp = te.compute(
        (batch_size, 1), lambda i, j: dp[i, j] * fc_w_dp[0, j], name="fc_dp"
    )

    fc_emb = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(wide[k2, i] * fc_w_emb[j, k2], axis=k2),
        name="fc_emb",
    )
    fc_out = topi.add(topi.add(fc_emb, fc_b), fc_dp)
    out = topi.nn.relu(fc_out)
    s = te.create_schedule(out.op)
    te.schedule.AutoInlineInjective(s)
    x, y = s[out].op.axis
    if batch_size >= 4:
        (m, n) = s[fc_emb].op.axis
        (k,) = s[fc_emb].op.reduce_axis
        # TUNABLE
        FACTOR = batch_size
        (mo, mi) = s[fc_emb].split(m, factor=FACTOR)
        s[fc_emb].reorder(mo, n, k, mi)
        s[fc_emb].vectorize(mi)
        (m, n) = s[dp].op.axis
        (k,) = s[dp].op.reduce_axis
        # TUNABLE
        (mo, mi) = s[dp].split(m, factor=FACTOR)
        s[dp].reorder(mo, n, k, mi)
        s[dp].vectorize(mi)

    func = tvm.build(
        s,
        [ad_emb, usr_emb, wide, fc_w_emb, fc_w_dp, fc_b, out],
        target=target,
        name="fused_op",
    )
    print(tvm.lower(s, [ad_emb, usr_emb, wide, fc_w_emb, fc_w_dp,  fc_b, out], simple_mode=True))
    # print(func.get_source('asm'))

    ad_emb_np = np.random.uniform(size=(emb_size, batch_size)).astype(ad_emb.dtype)
    usr_emb_np = np.random.uniform(size=(emb_size, 1)).astype(usr_emb.dtype)
    wide_np = np.random.uniform(size=(num_features, batch_size)).astype(wide.dtype)
    fc_w_emb_np = np.random.uniform(size=(1, num_features)).astype(fc_w_emb.dtype)
    fc_w_dp_np = np.random.uniform(size=(1, 1)).astype(fc_w_dp.dtype)

    fc_b_np = np.random.uniform(size=(1,)).astype(fc_b.dtype)
    out_np = np.zeros((batch_size, 1)).astype("float32")

    ctx = tvm.context(target, 0)
    ad_emb_nd = tvm.nd.array(ad_emb_np, ctx)
    usr_emb_nd = tvm.nd.array(usr_emb_np, ctx)
    wide_nd = tvm.nd.array(wide_np, ctx)
    fc_w_emb_nd = tvm.nd.array(fc_w_emb_np, ctx)
    fc_w_dp_nd = tvm.nd.array(fc_w_dp_np, ctx)

    fc_b_nd = tvm.nd.array(fc_b_np, ctx)
    out_nd = tvm.nd.array(out_np, ctx)

    func(ad_emb_nd, usr_emb_nd, wide_nd, fc_w_emb_nd, fc_w_dp_nd, fc_b_nd, out_nd)
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=100, repeat=3)
    mean = evaluator(
        ad_emb_nd, usr_emb_nd, wide_nd, fc_w_emb_nd, fc_w_dp_nd, fc_b_nd, out_nd
    ).mean
    flops = tvm.autotvm.task.task.compute_flop(s)
    print(
        "batch_size: ",
        batch_size,
        "time: ",
        mean * 1e6,
        " CPU per ad: ",
        mean / batch_size * 1e6,
        "us",
        "FLOPS per ad: ",
        flops / batch_size,
        " GFLOPS: ",
        flops / mean / 1e9,
    )


def deep_wide_dot(target, batch_size=128, num_features=50, emb_size=32):
    ad_emb = te.placeholder((batch_size, emb_size), name="ad_emb")
    usr_emb = te.placeholder((1, emb_size), name="usr_emb")
    wide = te.placeholder((batch_size, num_features), name="wide")

    fc_w = te.placeholder((emb_size, num_features), name="fc_w")
    fc_b = te.placeholder((emb_size,), name="fc_b")

    fc_w_linear = te.placeholder((1, 1), name="fc_w_linear")
    fc_b_linear = te.placeholder((1,), name="fc_b_linear")

    # Add + Mul, ReplaceNaN, Clip
    mu = te.placeholder((1, num_features), name="mu")
    sigma = te.placeholder((1, num_features), name="sigma")

    wide_normalized = topi.multiply(topi.add(wide, mu), sigma)
    wide_noNaN = te.compute(
        (batch_size, num_features),
        lambda i, j: tvm.tir.if_then_else(
            topi.isnan(wide_normalized)[i, j], 0, wide_normalized[i, j]
        ),
        name="wide_noNaN",
    )

    wide_preproc = topi.clip(wide_noNaN, -10.0, 10.0)

    # wide [batch_size, num_feature] => [batch_size, emb_size]
    k = te.reduce_axis((0, num_features), "k")
    wide_fc = te.compute(
        (batch_size, emb_size),
        lambda i, j: te.sum(wide_preproc[i, k] * fc_w[j, k], axis=k),
        name="fc",
    )
    wide_fc = topi.nn.relu(topi.add(wide_fc, fc_b))

    # batch_matmul + flatten => matmul
    k0 = te.reduce_axis((0, emb_size), "k0")
    k1 = te.reduce_axis((0, emb_size), "k1")
    k2 = te.reduce_axis((0, emb_size), "k2")
    dp0 = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(ad_emb[i, k0] * usr_emb[j, k0], axis=k0),
        name="dp0",
    )

    dp1 = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(ad_emb[i, k1] * wide_fc[i, k1], axis=k1),
        name="dp1",
    )

    dp2 = te.compute(
        (batch_size, 1),
        lambda i, j: te.sum(usr_emb[j, k2] * wide_fc[i, k2], axis=k2),
        name="dp2",
    )

    sum = topi.add(dp0, topi.add(dp1, dp2))
    linear = topi.nn.relu(topi.add(topi.multiply(sum, fc_w_linear), fc_b_linear))
    out = topi.sigmoid(linear)

    s = te.create_schedule(out.op)
    te.schedule.AutoInlineInjective(s)
    x, y = s[out].op.axis
    fused = s[out].fuse(x, y)

    # if batch_size > 1:
    #     # m is batch_size
    #     (m, n) = s[fc].op.axis
    #     (k,) = s[fc].op.reduce_axis
    #     # TUNABLE
    #     (mo, mi) = s[fc].split(m, factor=16)
    #     s[fc].reorder(mo, n, k, mi)
    #     (m, n) = s[dp].op.axis
    #     (k,) = s[dp].op.reduce_axis
    #     # TUNABLE
    #     (mo, mi) = s[dp].split(m, factor=16)
    #     s[dp].reorder(mo, n, k, mi)

    args_tvm = [
        ad_emb,
        usr_emb,
        wide,
        mu,
        sigma,
        fc_w,
        fc_b,
        fc_w_linear,
        fc_b_linear,
        out,
    ]

    func = tvm.build(s, args_tvm, target=target, name="fused_op")
    print(tvm.lower(s, args_tvm, simple_mode=True))
    # print(func.get_source('asm'))

    ad_emb_np = np.random.uniform(size=(batch_size, emb_size)).astype(ad_emb.dtype)
    usr_emb_np = np.random.uniform(size=(1, emb_size)).astype(usr_emb.dtype)
    wide_np = np.random.uniform(size=(batch_size, num_features)).astype(wide.dtype)
    mu_np = np.random.uniform(size=(1, num_features)).astype(mu.dtype)
    sigma_np = np.random.uniform(size=(1, num_features)).astype(sigma.dtype)
    fc_w_np = np.random.uniform(size=(emb_size, num_features)).astype(fc_w.dtype)
    fc_b_np = np.random.uniform(size=(emb_size,)).astype(fc_b.dtype)
    fc_w_linear_np = np.random.uniform(size=(1, 1)).astype(fc_w_linear.dtype)
    fc_b_linear_np = np.random.uniform(size=(1,)).astype(fc_b_linear.dtype)
    out_np = np.zeros((batch_size, 1)).astype("float32")

    ctx = tvm.context(target, 0)

    ad_emb_nd = tvm.nd.array(ad_emb_np, ctx)
    usr_emb_nd = tvm.nd.array(usr_emb_np, ctx)
    wide_nd = tvm.nd.array(wide_np, ctx)
    mu_nd = tvm.nd.array(mu_np, ctx)
    sigma_nd = tvm.nd.array(sigma_np, ctx)
    fc_w_nd = tvm.nd.array(fc_w_np, ctx)
    fc_b_nd = tvm.nd.array(fc_b_np, ctx)
    fc_w_linear_nd = tvm.nd.array(fc_w_linear_np, ctx)
    fc_b_linear_nd = tvm.nd.array(fc_b_linear_np, ctx)

    out_nd = tvm.nd.array(out_np, ctx)

    eval_args = (
        ad_emb_nd,
        usr_emb_nd,
        wide_nd,
        mu_nd,
        sigma_nd,
        fc_w_nd,
        fc_b_nd,
        fc_w_linear_nd,
        fc_b_linear_nd,
        out_nd,
    )

    func(*eval_args)
    evaluator = func.time_evaluator(func.entry_name, ctx, repeat=3, min_repeat_ms=100)
    mean = evaluator(*eval_args).mean
    flops = tvm.autotvm.task.task.compute_flop(s)
    print(
        "batch_size: ",
        batch_size,
        "time: ",
        mean * 1e6,
        ", CPU per ad: ",
        mean / batch_size * 1e6,
        "us, GFlops: ",
        flops / mean / 1e9,
    )


@click.command()
@click.option("--device", type=click.Choice(TARGETS.keys()))
@click.option("--net", type=click.Choice(["transpose", "split_concat", "dot", "std"]))
def run(device, net):
    if net == "transpose":
        deep_wide_transpose(TARGETS[device], 1)
        deep_wide_transpose(TARGETS[device], 4)
        deep_wide_transpose(TARGETS[device], 16)
        deep_wide_transpose(TARGETS[device], 64)
        deep_wide_transpose(TARGETS[device], 128)
        deep_wide_transpose(TARGETS[device], 512)
        deep_wide_transpose(TARGETS[device], 1024)
    elif net == "split_concat":
        deep_wide(TARGETS[device], 1)
        deep_wide(TARGETS[device], 4)
        deep_wide(TARGETS[device], 16)
        deep_wide(TARGETS[device], 64)
        deep_wide(TARGETS[device], 128)
        deep_wide(TARGETS[device], 512)
        deep_wide(TARGETS[device], 1024)
    elif net == "dot":
        deep_wide_dot(TARGETS[device], 1)
        # deep_wide_dot(TARGETS[device], 4)
        # deep_wide_dot(TARGETS[device], 16)
        # deep_wide_dot(TARGETS[device], 64)
        # deep_wide_dot(TARGETS[device], 128)
        # deep_wide_dot(TARGETS[device], 512)
        # deep_wide_dot(TARGETS[device], 1024)
    elif net == "std":
        deep_wide(TARGETS[device], 1, False)
        deep_wide(TARGETS[device], 4, False)
        deep_wide(TARGETS[device], 16, False)
        deep_wide(TARGETS[device], 64, False)
        deep_wide(TARGETS[device], 128, False)
        deep_wide(TARGETS[device], 512, False)
        deep_wide(TARGETS[device], 1024, False)


if __name__ == "__main__":
    run()
