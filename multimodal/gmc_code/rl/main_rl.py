import os
import torch
import collections
import collections.abc

from gmc_code.rl.modules.trainers.controller_trainer import count_trainable

collections.Mapping = collections.abc.Mapping
import sacred
import gmc_code.rl.ingredients.exp_ingredients as sacred_exp
import gmc_code.rl.ingredients.machine_ingredients as sacred_machine
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from gmc_code.rl.modules.trainers.model_trainer import ModelLearner
from gmc_code.rl.modules.sacred_loggers import SacredLogger
from gmc_code.rl.data_modules.envs.atari_env import AtariEnv
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.strategies import DDPStrategy
from torch.utils.flop_counter import FlopCounterMode
from gmc_code.rl.modules.callbacks import (
    OnEndModelTrainingPendulum,
    OnEndControllerTrainingPendulumPostEvalCb,
    OnControllerEvalPendulumPostEvalCb,
)
from gmc_code.rl.utils.general_utils import (
    setup_dca_evaluation_trainer,
    setup_model,
    setup_data_module,
    load_model,
    setup_downstream_controller,
    setup_downstream_controller_trainer,
    load_down_controller,
)
import time

AVAIL_GPUS = min(1, torch.cuda.device_count())

ex = sacred.Experiment(
    "GMC_experiment_rl",
    ingredients=[sacred_machine.machine_ingredient, sacred_exp.exp_ingredient],
)


@ex.capture
def log_dir_path(folder, _config, _run):
    model_type = str(_config["experiment"]["model"])
    exp_name = str(_config["experiment"]["scenario"])

    return os.path.join(
        _config["machine"]["m_path"],
        "evaluation/",
        model_type + "_" + exp_name,
        f'log_{_config["experiment"]["seed"]}',
        folder,
    )


@ex.capture
def trained_model_dir_path(file, _config, _run):
    return os.path.join(
        _config["machine"]["m_path"],
        "trained_models/",
        file
    )


@ex.capture
def load_hyperparameters(_config, _run):
    exp_cfg = _config["experiment"]
    scenario_cfg = _config["experiment"]["scenario_config"]
    model_cfg = _config["experiment"]["model_config"]

    return exp_cfg, scenario_cfg, model_cfg


@ex.capture
def train_model(_config, _run):
    # Init model input parameters
    exp_cfg, scenario_cfg, model_cfg = load_hyperparameters()
    model_train_cfg = _config["experiment"]["model_train_config"]

    # Set the seeds
    seed_everything(exp_cfg["seed"], workers=True)

    # Init model
    model = setup_model(
        model=exp_cfg["model"],
        scenario=exp_cfg["scenario"],
        scenario_config=scenario_cfg,
        model_config=model_cfg,
    )

    # Init Trainer
    model_trainer = ModelLearner(
        model=model,
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
        model_name=exp_cfg["model"],
    )

    # Init Data Module
    data_module = setup_data_module(
        scenario=exp_cfg["scenario"],
        experiment_config=exp_cfg,
        scenario_config=scenario_cfg,
        train_config=model_train_cfg,
    )

    # Loggers
    sacred_logger = SacredLogger(sacred_experiment=ex)

    # Train
    checkpoint_dir = log_dir_path("checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}-"
                 + "{epoch:02d}",
        monitor="val_weighted_loss",
        every_n_epochs=model_train_cfg["snapshot"],
        save_top_k=-1,
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = (
        f"{exp_cfg['model']}_{exp_cfg['scenario']}_{sacred_logger.run_id}_last"
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Callbacks
    if exp_cfg["scenario"] == "pendulum" or exp_cfg["scenario"] == "pendulum2" or exp_cfg["scenario"] == "pendulum3" or exp_cfg["scenario"] == "pendulum4":
        end_callback = OnEndModelTrainingPendulum(exp_cfg["scenario"])
    elif exp_cfg["scenario"] in ("hyperhot", "hyperhot2", "hyperhot3", "hyperhot4","hyperhot5", "hyperhot6", "hyperhot7", "hyperhot8", "hyperhot9"):
        end_callback = OnEndModelTrainingPendulum(exp_cfg["scenario"])
    elif "vizdoom" in exp_cfg["scenario"]:
        end_callback = OnEndModelTrainingPendulum(exp_cfg["scenario"])
    else:
        raise ValueError("Error")
    checkpoint_dir = log_dir_path("checkpoints")
    tensorboard_logger = TensorBoardLogger(
        "trained_models/evaluation/train_model/", name=exp_cfg["scenario"]
    )
    # TEST_limit_train_batches = 0.01
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=model_train_cfg["epochs"],
        enable_progress_bar=True,
        default_root_dir=log_dir_path("saved_models"),
        logger=[sacred_logger, tensorboard_logger],
        callbacks=[checkpoint_callback, end_callback],
    )

    # Train
    trainer.fit(model_trainer, data_module)

@ex.main
def main(_config, _run):
    start = time.perf_counter()
    # Run experiment
    if _config["experiment"]["stage"] == "train_model":
        os.makedirs(log_dir_path("saved_models"), exist_ok=True)
        os.makedirs(log_dir_path("checkpoints"), exist_ok=True)
        train_model()
    else:
        raise ValueError(
            "[RL Experiment] Incorrect stage of pipeline selected: "
            + str(_config["experiment"]["stage"])
        )
    end = time.perf_counter()
    print(f"Execution time: {end - start:.6f} seconds")


if __name__ == "__main__":
    ex.run_commandline()
