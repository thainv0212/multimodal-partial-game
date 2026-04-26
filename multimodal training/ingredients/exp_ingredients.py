import sacred
import gmc_code.rl.ingredients.stage_specific.model_ingredients as sacred_model
import gmc_code.rl.ingredients.stage_specific.scenario_ingredients as sacred_scenario
import gmc_code.rl.ingredients.stage_specific.down_ingredients as sacred_down
import gmc_code.rl.ingredients.stage_specific.dca_evaluation_ingredients as sacred_dca

########################
#     Experiment       #
########################

exp_ingredient = sacred.Ingredient("experiment")


@exp_ingredient.config
def exp_config():
    # Experiment setup
    scenario = "pendulum"
    model = "gmc"
    seed = 0
    cuda = True

    # Experiment id (for checkpoints)
    exp_id = None

    # Stages
    # Model Training        - 'train_model'
    # Model Evaluation      - 'evaluate_dca',
    # Downstream Training   - 'train_downstream_controller'
    # Downstream Evaluation - 'evaluate_downstream_controller'

    stage = "train_model"
    evaluation_mods = [0, 1]

    # Load model and scenario specific ingredients
    if scenario == "pendulum":
        scenario_config = sacred_scenario.pendulum()
        down_train_config = sacred_down.pendulum()
        down_eval_config = sacred_down.pendulum_eval()
        dca_evaluation_config = sacred_dca.pendulum()
        model_config = sacred_model.gmc_pendulum()
        model_train_config = sacred_model.gmc_pendulum_train()
    if scenario == "pendulum2":
        scenario_config = sacred_scenario.pendulum2()
        down_train_config = sacred_down.pendulum()
        down_eval_config = sacred_down.pendulum_eval()
        dca_evaluation_config = sacred_dca.pendulum()
        model_config = sacred_model.gmc_pendulum()
        model_train_config = sacred_model.gmc_pendulum_train()
    if scenario == "pendulum3":
        scenario_config = sacred_scenario.pendulum3()
        down_train_config = sacred_down.pendulum()
        down_eval_config = sacred_down.pendulum_eval()
        dca_evaluation_config = sacred_dca.pendulum()
        model_config = sacred_model.gmc_pendulum()
        model_train_config = sacred_model.gmc_pendulum_train()
    if scenario == "pendulum4":
        scenario_config = sacred_scenario.pendulum4()
        down_train_config = sacred_down.pendulum()
        down_eval_config = sacred_down.pendulum_eval()
        dca_evaluation_config = sacred_dca.pendulum()
        model_config = sacred_model.gmc_pendulum()
        model_train_config = sacred_model.gmc_pendulum_train()
    if scenario == "hyperhot":
        scenario_config = sacred_scenario.hyperhot()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot2":
        scenario_config = sacred_scenario.hyperhot2()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot3":
        scenario_config = sacred_scenario.hyperhot3()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot4":
        scenario_config = sacred_scenario.hyperhot4()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot5":
        scenario_config = sacred_scenario.hyperhot5()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot6":
        scenario_config = sacred_scenario.hyperhot6()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot7":
        scenario_config = sacred_scenario.hyperhot7()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot8":
        scenario_config = sacred_scenario.hyperhot8()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "hyperhot9":
        scenario_config = sacred_scenario.hyperhot9()
        model_config = sacred_model.gmc_hyperhot()
        model_train_config = sacred_model.gmc_hyperhot_train()
    if scenario == "vizdoom0":
        scenario_config = sacred_scenario.vizdoom0()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom1":
        scenario_config = sacred_scenario.vizdoom1()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom2":
        scenario_config = sacred_scenario.vizdoom2()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom3":
        scenario_config = sacred_scenario.vizdoom3()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom4":
        scenario_config = sacred_scenario.vizdoom4()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom5":
        scenario_config = sacred_scenario.vizdoom5()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()

    if scenario == "vizdoom6":
        scenario_config = sacred_scenario.vizdoom6()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()

    if scenario == "vizdoom7":
        scenario_config = sacred_scenario.vizdoom7()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom8":
        scenario_config = sacred_scenario.vizdoom8()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom9":
        scenario_config = sacred_scenario.vizdoom9()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom10":
        scenario_config = sacred_scenario.vizdoom10()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom11":
        scenario_config = sacred_scenario.vizdoom11()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom12":
        scenario_config = sacred_scenario.vizdoom12()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom13":
        scenario_config = sacred_scenario.vizdoom13()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom14":
        scenario_config = sacred_scenario.vizdoom14()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom15":
        scenario_config = sacred_scenario.vizdoom15()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom16":
        scenario_config = sacred_scenario.vizdoom16()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom17":
        scenario_config = sacred_scenario.vizdoom17()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom18":
        scenario_config = sacred_scenario.vizdoom18()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom19":
        scenario_config = sacred_scenario.vizdoom19()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()
    if scenario == "vizdoom20":
        scenario_config = sacred_scenario.vizdoom20()
        model_config = sacred_model.gmc_vizdoom()
        model_train_config = sacred_model.gmc_vizdoom_train()