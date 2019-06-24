from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import time

import _pickle as pickle
import click
import numpy as np
import tvm
import tvm.contrib.graph_runtime
from tvm import autotvm, relay

from tvm.relay.ir_pass import *

import dense
import batch_matmul


TARGETS = dict(
    skl="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu",
    rtp="llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu",
    dev="llvm -mcpu=skylake -target=x86_64-linux-gnu",
    mac="llvm -mcpu=core-avx2",
)

def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=2000,
    early_stopping=100,
    log_filename="tuning.log",
    use_transfer_learning=False,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        print(tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = autotvm.tuner.XGBTuner(
                tsk, loss_type="rank", feature_type="knob"
            )
        elif tuner == "ga":
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        print(tsk.config_space)
        tuner_obj.tune(
            n_trial=min(n_trial, len(tsk.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


@click.command()
@click.option("--func_pkl", type=click.Path())
@click.option("--params_pkl", type=click.Path())
@click.option("--inputs_pkl", type=click.Path())
@click.option("--autotvm_number", default=50)
@click.option("--autotvm_repeat", default=4)
@click.option("--autotvm_n_trial", default=200)
@click.option("--autotvm_early_stopping", default=100)
@click.option("--autotvm_log", default="sparse_tuning.log", type=click.Path())
@click.option("--tracker_port", default=9195)
@click.option("--device", type=click.Choice(TARGETS.keys()))
@click.option("--opt_level", default=2)
def run(
    func_pkl,
    params_pkl,
    inputs_pkl,
    autotvm_number,
    autotvm_repeat,
    autotvm_log,
    autotvm_n_trial,
    autotvm_early_stopping,
    tracker_port,
    device,
    opt_level,
):
    target = TARGETS[device]
    logging.basicConfig(level=logging.DEBUG)

    with open(func_pkl, "rb") as f:
        func = pickle.load(f)

    with open(params_pkl, "rb") as f:
        params = pickle.load(f)

    with open(inputs_pkl, "rb") as f:
        inputs = pickle.load(f)

    func = infer_type(func)
    print(func)

    with relay.build_config(opt_level=opt_level):
        tasks = autotvm.task.extract_from_program(
            func,
            target=target,
            params=params,
            ops=[
                relay.op.nn.dense,
                relay.op.nn.batch_matmul,
            ],
        )

    for i, task in enumerate(tasks):
        logging.info("Task %s: %s", i, task)

    tune_tasks(
        tasks,
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=50)
            if device == "skl"
            else autotvm.LocalBuilder(timeout=50, n_parallel=1),
            runner=autotvm.RPCRunner(
                "skl",
                "localhost",
                tracker_port,
                number=autotvm_number,
                repeat=autotvm_repeat,
                timeout=50,
            )
            if device == "skl"
            else autotvm.LocalRunner(
                number=autotvm_number, repeat=autotvm_repeat, timeout=50
            ),
        ),
        n_trial=autotvm_n_trial,
        early_stopping=autotvm_early_stopping,
        log_filename=str(autotvm_log),
    )


if __name__ == "__main__":
    run()
