import sacred

###########################
#        Model            #
###########################

model_ingredient = sacred.Ingredient("model")


# Pendulum

@model_ingredient.config
def gmc_pendulum():
    model = "gmc"
    common_dim = 64
    latent_dim = 10
    loss_type = "prepared_for_ablation"  # "joints_as_negatives"
    learner = 'ddpg'
    model_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/pendulum/rl-sound-only-design-2-[\"LEFT_TOP\", \"RIGHT_TOP\", \"MIDDLE_BOTTOM\"]/down_sound_pendulum2_model.pth.tar"


##############################
#       Model  Train         #
##############################


model_train_ingredient = sacred.Ingredient("model_train")


@model_train_ingredient.named_config
def gmc_pendulum_train():
    # Dataset parameters
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 500
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    temperature = 0.3

@model_ingredient.config
def gmc_hyperhot():
    model = "gmc"
    common_dim = 64
    latent_dim = 40
    loss_type = "infonce"  # "joints_as_negatives"
    learner = 'ddpg'
    model_file = "/mnt/DATA/Research/Laboratory/Research/gmc_no_modification/gmc_code/rl/trained_models/finetune-design 1/down_gmc_pendulum_model.pth.tar"


@model_train_ingredient.named_config
def gmc_hyperhot_train():
    # Dataset parameters
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 250
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    temperature = 0.3

@model_ingredient.config
def gmc_vizdoom():
    model = "gmc"
    common_dim = 64
    latent_dim = 40
    loss_type = "infonce"  # "joints_as_negatives"

@model_train_ingredient.named_config
def gmc_vizdoom_train():
    # Dataset parameters
    batch_size = 128
    num_workers = 8

    # Training Hyperparameters
    epochs = 250
    learning_rate = 1e-3
    snapshot = 50
    checkpoint = None

    temperature = 0.3