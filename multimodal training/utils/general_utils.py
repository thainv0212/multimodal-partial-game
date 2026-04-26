# from gmc_code.rl.architectures.models.gmc import PendulumGMC, PendulumNullModel, HyperhotGMC, VizdoomGMC, VizdoomGMC2, VizdoomGMC3, VizdoomGMC4, VizdoomGMC5, VizdoomGMC6, VizdoomGMC7, VizdoomGMC8, VizdoomGMC9, VizdoomGMC10, VizdoomGMC11, VizdoomGMC12
from gmc_code.rl.architectures.models.gmc import *
from gmc_code.rl.architectures.downstream.ddpg import DDPG, DDPG2, DDPG3
from gmc_code.rl.modules.trainers.controller_trainer import DDPGLearner
from gmc_code.rl.modules.trainers.dca_evaluation_trainer import DCAEvaluator
from gmc_code.rl.data_modules.DCAMultiAtari_dataset import *
from gmc_code.rl.data_modules.MultiAtari_dataset import MultiAtariDataModule, MultiHyperhotDataModule, MultiVizdoomData2Module, \
    MultiVizdoomDataModule, MultiVizdoomData3Module

"""

Setup functions

"""


def setup_model(scenario, model, model_config, scenario_config, finetune=False):
    if model == "gmc":
        if scenario == "pendulum" or scenario == "pendulum2" or scenario == "pendulum3"or scenario == "pendulum4":
            return PendulumGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
        if "hyperhot" in scenario:
            return HyperhotGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
        elif "vizdoom" in scenario:
            return VizdoomGMC(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for GMC Single model: "
                + str(scenario)
            )
    if model == "sound":
        if scenario == "pendulum" or scenario == "pendulum2" or scenario == "pendulum3" or scenario == "pendulum4":
            return PendulumNullModel()
        else:
            raise ValueError(
                "[Model Setup] Selected scenario not yet implemented for sound only model: "
                + str(scenario)
            )
    if model == "gmc2":
        if "vizdoom" in scenario:
            return VizdoomGMC2(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc3":
        if "vizdoom" in scenario:
            return VizdoomGMC3(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc4":
        if "vizdoom" in scenario:
            return VizdoomGMC4(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc5":
        if "vizdoom" in scenario:
            return VizdoomGMC5(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc6":
        if "vizdoom" in scenario:
            return VizdoomGMC6(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc7":
        if "vizdoom" in scenario:
            return VizdoomGMC7(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc8":
        if "vizdoom" in scenario:
            return VizdoomGMC8(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc9":
        if "vizdoom" in scenario:
            return VizdoomGMC9(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc10":
        if "vizdoom" in scenario:
            return VizdoomGMC10(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc11":
        if "vizdoom" in scenario:
            return VizdoomGMC11(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc12":
        return VizdoomGMC12(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc13":
        return VizdoomGMC13(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc14":
        return VizdoomGMC14(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc15":
        return VizdoomGMC15(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    if model == "gmc16":
        return VizdoomGMC16(
                name=model_config["model"],
                common_dim=model_config["common_dim"],
                latent_dim=model_config["latent_dim"],
                loss_type=model_config["loss_type"],
                finetune=None if finetune == False else scenario_config["finetune"],
            )
    else:
        raise ValueError(
            "[Model Setup] Selected model not yet implemented: " + str(model)
        )

def setup_data_module(scenario, experiment_config, scenario_config, train_config):
    if experiment_config["stage"] == "evaluate_dca":
        return DCAMultiAtariDataModule(
            dataset=scenario,
            scenario_config=scenario_config,
            data_dir=scenario_config["data_dir"],
            data_config=train_config,
        )

    elif scenario == "pendulum" or scenario == "pendulum2" or scenario == "pendulum3" or scenario == "pendulum4":
        return MultiAtariDataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            scenario_config=scenario_config,
            data_config=train_config,
        )
    elif scenario in ("hyperhot", "hyperhot2", "hyperhot3", "hyperhot4", "hyperhot5", "hyperhot6", "hyperhot7", "hyperhot8", "hyperhot9"):
        return MultiHyperhotDataModule(
            dataset=scenario,
            data_dir=scenario_config["data_dir"],
            scenario_config=scenario_config,
            data_config=train_config, )
    elif "vizdoom" in scenario:
        if experiment_config["model"] == "gmc":
            return MultiVizdoomDataModule(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                scenario_config=scenario_config,
                data_config=train_config,
            )
        elif experiment_config["model"] == "gmc2":
            return MultiVizdoomData2Module(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                scenario_config=scenario_config,
                data_config=train_config
            )
        elif experiment_config["model"] == "gmc3":
            return MultiVizdoomData2Module(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                scenario_config=scenario_config,
                data_config=train_config
            )
        # elif experiment_config["model"] in ("gmc4", "gmc5", "gmc6", "gmc7", "gmc8", "gmc9"):
        elif "gmc" in experiment_config["model"]:
            return MultiVizdoomData3Module(
                dataset=scenario,
                data_dir=scenario_config["data_dir"],
                scenario_config=scenario_config,
                data_config=train_config
            )
    else:
        raise ValueError(
            "[Data Module Setup] Selected Module not yet implemented: " + str(scenario)
        )


def setup_dca_evaluation_trainer(model, machine_path, scenario, config):
    return DCAEvaluator(
        model=model,
        scenario=scenario,
        machine_path=machine_path,
        minimum_cluster_size=config["minimum_cluster_size"],
        unique_modality_idxs=config["unique_modality_idxs"],
        unique_modality_dims=config["unique_modality_dims"],
        partial_modalities_idxs=config["partial_modalities_idxs"],
    )


def setup_downstream_controller(scenario, model_config, n_actions, layer_sizes):
    if scenario == "pendulum" or scenario == "pendulum2" or scenario == "pendulum3" or scenario == "pendulum4":
        if model_config["learner"] == "ddpg":
            return DDPG(
                model_config["latent_dim"], n_actions=n_actions, layer_sizes=layer_sizes
            )
        elif model_config["learner"] == "ddpg2":
            return DDPG2(
                model_config["latent_dim"], n_actions=n_actions, layer_sizes=layer_sizes
            )
        elif model_config["learner"] == "ddpg3":
            return DDPG3(
                model_config["latent_dim"], n_actions=n_actions, layer_sizes=layer_sizes,
                model_file=model_config["model_file"]
            )
        else:
            raise ValueError("Not implemented learner: " + str(model_config["learner"]))
    else:
        raise ValueError(
            "[Down Controller Setup] Selected scenario not yet implemented: "
            + str(scenario)
        )


def setup_downstream_controller_trainer(
        scenario, model, controller, env, scenario_config, train_config, logger, modalities, cuda=False
):
    if scenario == "pendulum" or scenario == "pendulum2" or scenario == "pendulum3" or scenario == "pendulum4":  # Classification
        return DDPGLearner(
            model=model,
            controller=controller,
            env=env,
            scenario_config=scenario_config,
            train_config=train_config,
            logger=logger,
            modalities=modalities,
            cuda=cuda
        )
    else:
        raise ValueError(
            "[Down Controller Trainer Setup] Trainer for selected scenario not yet implemented: "
            + str(scenario)
        )


"""

Loading functions

"""


def load_model(sacred_config, model_file):
    model = setup_model(
        scenario=sacred_config["experiment"]["scenario"],
        model=sacred_config["experiment"]["model"],
        model_config=sacred_config["experiment"]["model_config"],
        scenario_config=sacred_config["experiment"]["scenario_config"],
    )
    if model_file is not None:
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    model.freeze()
    if sacred_config['experiment']['cuda']:
        model.cuda()

    return model


def load_down_controller(sacred_config, down_model_file, n_actions, layer_sizes):
    down_model = setup_downstream_controller(
        scenario=sacred_config["experiment"]["scenario"],
        model_config=sacred_config["experiment"]["model_config"],
        n_actions=n_actions,
        layer_sizes=layer_sizes,
    )

    checkpoint = torch.load(down_model_file)
    down_model.load_state_dict(checkpoint["state_dict"])

    # Freeze model
    down_model.freeze()

    return down_model


"""


General functions

"""


def flatten_dict(dd, separator="_", prefix=""):
    return (
        {
            prefix + separator + k if prefix else k: v
            for kk, vv in dd.items()
            for k, v in flatten_dict(vv, separator, kk).items()
        }
        if isinstance(dd, dict)
        else {prefix: dd}
    )
