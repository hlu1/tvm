from __future__ import absolute_import, print_function

import logging
import click
import numpy as np
import topi
import tvm
from tvm import relay
from tvm.contrib import graph_runtime
from tvm.relay.testing import create_workload


TARGETS = dict(
    skl="llvm -mcpu=skylake -target=x86_64-linux-gnu", #avx2
    skl_512="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu", #avx512
    brw="llvm -mcpu=broadwell -target=x86_64-linux-gnu", #avx2
)


def get_model(batch_size, num_features, emb_size):
    # inputs
    ad_emb = relay.var("ad_emb", shape=(batch_size, emb_size))
    usr_emb = relay.var("usr_emb", shape=(1, emb_size))
    wide = relay.var("wide", shape=(batch_size, num_features))

    # weights/params
    fc_w = relay.var("fc_w", shape=(1, num_features + 1))
    fc_b = relay.var("fc_b", shape=(1,))
    mu = relay.var("mu", shape=(1, num_features))
    sigma = relay.var("sigma", shape=(1, num_features))

    # build graph
    # Add + Mul, ReplaceNaN, Clip
    wide_normalized = relay.multiply(relay.add(wide, mu), sigma)
    wide_noNaN = relay.where(relay.isnan(wide_normalized), relay.zeros_like(wide_normalized), wide_normalized)
    wide_preproc = relay.clip(wide_noNaN, -10.0, 10.0)

    dp = relay.nn.dense(ad_emb, usr_emb) # FC
    output = relay.concatenate([dp, wide_preproc], axis=1)
    fc = relay.nn.dense(output, fc_w)
    out = relay.sigmoid(relay.add(fc, fc_b))

    func = relay.expr.Function(relay.analysis.free_vars(out), out)
    mod = tvm.IRModule.from_expr(func)

    # initialize weights/params
    params = {}
    fc_w_np = np.random.uniform(size=(1, num_features + 1)).astype(np.float32)
    fc_b_np = np.random.uniform(size=(1,)).astype(np.float32)
    mu_np =np.random.uniform(size=(1, num_features)).astype(np.float32)
    sigma_np =np.random.uniform(size=(1, num_features)).astype(np.float32)
    params["fc_w"] = tvm.nd.array(fc_w_np, ctx=tvm.cpu(0))
    params["fc_b"] = tvm.nd.array(fc_b_np, ctx=tvm.cpu(0))
    params["mu"] = tvm.nd.array(mu_np, ctx=tvm.cpu(0))
    params["sigma"] = tvm.nd.array(sigma_np, ctx=tvm.cpu(0))

    return mod, params


def deep_wide_relay(target, batch_size=128, num_features=50, emb_size=32):
    mod, params = get_model(batch_size, num_features, emb_size)
    # print(mod)
    # print(params)

    ctx = tvm.context(target, 0)
    graph, lib, params = relay.build(mod, target, params=params)
    mod = graph_runtime.create(graph, lib, ctx=ctx)
    mod.set_input(**params)

    ad_emb_np = np.random.uniform(size=(batch_size, emb_size)).astype(np.float32)
    usr_emb_np = np.random.uniform(size=(1, emb_size)).astype(np.float32)
    wide_np = np.random.uniform(size=(batch_size, num_features)).astype(np.float32)

    ad_emb_nd = tvm.nd.array(ad_emb_np, ctx)
    usr_emb_nd = tvm.nd.array(usr_emb_np, ctx)
    wide_nd = tvm.nd.array(wide_np, ctx)
    inputs = {"ad_emb": ad_emb_nd, "usr_emb": usr_emb_nd, "wide": wide_nd}
    mod.set_input(**inputs)

    mod.run()
    evaluator = mod.module.time_evaluator("run", ctx, repeat=5, min_repeat_ms=100)
    mean = evaluator().mean
    print(
        "batch_size: ",
        batch_size,
        "time: ",
        mean * 1e6,
        ", CPU per ad: ",
        mean / batch_size * 1e6,
    )


@click.command()
@click.option("--device", type=click.Choice(TARGETS.keys()))
@click.option("--verbose", is_flag=True)
def run(device, verbose):
    logging.basicConfig(level=logging.ERROR if not verbose else logging.DEBUG)
    target = TARGETS[device]
    deep_wide_relay(target, 1)
    deep_wide_relay(target, 4)
    deep_wide_relay(target, 16)
    deep_wide_relay(target, 64)


if __name__ == "__main__":
    run()
