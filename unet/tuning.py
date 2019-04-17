from tvm import autotvm
import click
import logging
import nnvm
import nnvm.compiler
from nnvm.frontend import from_caffe2
import numpy as np
import os
import tvm

from collections import OrderedDict
from caffe2.python import workspace
from caffe2.proto import caffe2_pb2, metanet_pb2

skl_target = tvm.target.create("llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu")
local_target = tvm.target.create("llvm -mcpu=core-avx2")
rpi_target = tvm.target.arm_cpu("rasp3b")


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


def apply_template(tasks, template):
    def f(task):
        try:
            return autotvm.task.create(
                task.name, task.args, task.target, task.target_host, template
            )
        except (AssertionError, KeyError):
            logging.exception(
                "Failed to construct Winograd variant task for task: %s", task
            )
            return None

    ts = [f(task) for task in tasks]
    return [t for t in ts if t is not None]


@click.command()
@click.option("--init_net", type=click.Path())
@click.option("--pred_net", type=click.Path())
@click.option("--input_size1", type=(int, int, int, int), default=(1, 3, 192, 192))
@click.option("--input_size2", type=(int, int, int, int), default=(1, 4, 192, 192))
@click.option("--autotvm_number", default=50)
@click.option("--autotvm_repeat", default=4)
@click.option("--autotvm_n_trial", default=200)
@click.option("--autotvm_early_stopping", default=100)
@click.option("--autotvm_log", type=click.Path())
@click.option("--tracker_port", default=9195)
@click.option(
    "--template",
    type=click.Choice(
        ["winograd_nnpack_fp16", "winograd_nnpack_fp32", "winograd", "direct"]
    ),
    required=True,
)
@click.option("--opt_level", default=2)
@click.option("--device", type=click.Choice(["skl", "rpi", "local"]), required=True)
def run(
    init_net,
    pred_net,
    input_size1,
    input_size2,
    autotvm_number,
    autotvm_repeat,
    autotvm_n_trial,
    autotvm_early_stopping,
    autotvm_log,
    tracker_port,
    template,
    opt_level,
    device,
):
    logging.basicConfig(level=logging.DEBUG)
    target = dict(skl=skl_target, rpi=rpi_target, local=local_target)[device]

    with open(init_net, "rb") as f:
        init_net = metanet_pb2.MetaNetDef()
        init_net.ParseFromString(f.read())
    with open(pred_net, "rb") as f:
        pred_net = metanet_pb2.MetaNetDef()
        pred_net.ParseFromString(f.read())

    num_models = len(pred_net.nets)
    assert num_models <= 2
    if num_models == 2:
        input_shapes = [input_size1, input_size2]
    else:
        input_shapes = [input_size1]
    all_tasks = OrderedDict()
    for index in range(num_models):
        init = init_net.nets[index].value
        pred = pred_net.nets[index].value
        input_name = pred.external_input[0]
        sym, params = from_caffe2(init, pred)
        assert params

        with nnvm.compiler.build_config(opt_level=opt_level):
            graph, lib, new_params = nnvm.compiler.build(
                sym, target, shape={input_name: input_shapes[index]}, params=params
            )

        shape = {k: v.shape for k, v in params.items()}
        shape[input_name] = input_shapes[index]
        with nnvm.compiler.build_config(opt_level=opt_level):
            tasks = autotvm.task.extract_from_graph(
                sym,
                shape=shape,
                dtype="float32",
                target=target,
                symbols=[nnvm.sym.conv2d, nnvm.sym.contrib.conv2d_NCHWc],
            )
        for task in tasks:
            all_tasks[task.workload] = task

    tasks = [task for w, task in all_tasks.items()]
    tasks = apply_template(tasks, template)

    for i, task in enumerate(tasks):
        logging.info("Task %s: %s", i, task)
    tune_tasks(
        tasks,
        measure_option=autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=50),
            runner=autotvm.RPCRunner(
                device,
                "localhost",
                tracker_port,
                number=autotvm_number,
                repeat=autotvm_repeat,
                timeout=50,
            ),
        ),
        n_trial=autotvm_n_trial,
        early_stopping=autotvm_early_stopping,
        log_filename=str(autotvm_log),
    )


if __name__ == "__main__":
    run()
