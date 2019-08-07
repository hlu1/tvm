from tvm import autotvm
import tvm.contrib.graph_runtime
import click
import logging
import nnvm
import nnvm.compiler
import numpy as np
import os
import time
import tvm

import models
# import unet_conv2d

skl_target = tvm.target.create('llvm -mcpu=skylake-avx512 -target=x86_64-linux-gnu')
local_target = tvm.target.create('llvm -mcpu=core-avx2')
rpi_target = tvm.target.arm_cpu("rasp3b")


# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=2000,
               early_stopping=100,
               log_filename='tuning.log',
               use_transfer_learning=False):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(tasks):
        print(tsk)
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = autotvm.tuner.XGBTuner(tsk, loss_type='rank', feature_type="knob")
        elif tuner == 'ga':
            tuner_obj = autotvm.tuner.GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = autotvm.tuner.RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = autotvm.tuner.GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        print(tsk.config_space)
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)

def apply_template(tasks, template):
    def f(task):
        try:
            return autotvm.task.create(task.name, task.args,
                                       task.target, task.target_host, template)
        except (AssertionError, KeyError):
            logging.exception("Failed to construct Winograd variant task for task: %s", task)
            return None
    ts = [f(task) for task in tasks]
    return [t for t in ts if t is not None]

@click.command()
@click.option('--align', default=1)
@click.option('--model', type=click.Choice(['unet', 'unet_track', 'resnet50', 'unet_le', 'unet_le_track']), required=True)
@click.option('--autotvm_number', default=50)
@click.option('--autotvm_repeat', default=4)
@click.option('--autotvm_n_trial', default=200)
@click.option('--autotvm_early_stopping', default=100)
@click.option('--autotvm_log', default="autotvm_unet_tuning.log", type=str)
@click.option('--tracker_port', default=9195)
@click.option('--template',
              type=click.Choice(
                  ['winograd_nnpack', 'winograd', 'direct', 'direct_vanilla']), required=True)
@click.option('--opt_level', default=3)
@click.option('--device', type=click.Choice(["skl", "rpi", "local"]), required=True)
def run(align,
        model,
        autotvm_number,
        autotvm_repeat,
        autotvm_log,
        autotvm_n_trial,
        autotvm_early_stopping,
        tracker_port,
        template,
        opt_level,
        device):
    # logging.basicConfig(level=logging.DEBUG)
    target = dict(skl=skl_target, rpi=rpi_target, local=local_target)[device]

    sym, image_shape, output_shape = models.get_mxnet_symbol(model, align)
    sym, params = models.get_nnvm_sym(sym, image_shape)
    assert params

    data_shape = tuple([1] + list(image_shape))
    with nnvm.compiler.build_config(opt_level=opt_level):
        graph, lib, params = nnvm.compiler.build(sym, target, dict(data=data_shape), params=params)
    print("Succesfully built")
    print(graph.symbol().debug_str())
    with nnvm.compiler.build_config(opt_level=opt_level):
        tasks = autotvm.task.extract_from_graph(
            sym,
            shape=dict(data=data_shape),
            dtype="float32",
            target=target,
            symbols=[
                nnvm.sym.conv2d,
                nnvm.sym.contrib.conv2d_NCHWc,
            ]
        )
    tasks = apply_template(tasks, template)
    # tasks = [tasks[0]]

    for i, task in enumerate(tasks):
        logging.info("Task %s: %s", i, task)
    tune_tasks(tasks,
               measure_option=autotvm.measure_option(
                   builder=autotvm.LocalBuilder(timeout=50),
                   runner=autotvm.RPCRunner(
                       device, 'localhost', tracker_port,
                       number=autotvm_number,
                       repeat=autotvm_repeat,
                       timeout=50)
               ),
               n_trial=autotvm_n_trial,
               early_stopping=autotvm_early_stopping,
               log_filename=str(autotvm_log))

if __name__ == '__main__':
    run()
