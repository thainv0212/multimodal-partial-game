import sacred

########################
#      Scenario        #
########################

scenario_ingredient = sacred.Ingredient("scenario")


@scenario_ingredient.named_config
def pendulum():
    scenario = 'pendulum'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    # sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0


@scenario_ingredient.named_config
def pendulum2():
    scenario = 'pendulum2'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    # sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0

@scenario_ingredient.named_config
def pendulum3():
    scenario = 'pendulum3'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    # sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0

@scenario_ingredient.named_config
def pendulum4():
    scenario = 'pendulum4'
    data_dir = './dataset/'
    image_side = 60
    n_stack = 2
    sound_frequency = 440.
    sound_velocity = 20.
    # sound_receivers = ['LEFT_BOTTOM', 'RIGHT_BOTTOM', 'MIDDLE_TOP']
    sound_receivers = ["LEFT_TOP", "RIGHT_TOP", "MIDDLE_BOTTOM"]

    train_samples = 20000
    test_samples = 2000
    random_seed = 42
    finetune = "sound"
    train_vae = 0

@scenario_ingredient.named_config
def hyperhot():
    scenario = "hyperhot"
    data_dir='./dataset/hyperhot'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "LEFT_SHIP", "RIGHT_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot2():
    scenario = "hyperhot2"
    data_dir='./dataset/hyperhot2'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot3():
    scenario = "hyperhot3"
    data_dir='./dataset/hyperhot3'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot4():
    scenario = "hyperhot4"
    data_dir='./dataset/hyperhot4'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot5():
    scenario = "hyperhot5"
    data_dir='./dataset/hyperhot5'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot6():
    scenario = "hyperhot6"
    data_dir='./dataset/hyperhot6'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "RIGHT_BOTTOM", "TOP_SHIP", "BOTTOM_SHIP"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot7():
    scenario = "hyperhot7"
    data_dir='./dataset/hyperhot7'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot8():
    scenario = "hyperhot8"
    data_dir='./dataset/hyperhot8'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def hyperhot9():
    scenario = "hyperhot9"
    data_dir='./dataset/hyperhot9'
    train_samples = 32000
    test_samples = 8000
    image_side = 80
    sound_samples = 1047
    n_stack = 2
    n_enemies = 4
    pacifist_mode = False
    time_limit = 15
    sound_receivers = ["LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM", "LEFT_BOTTOM"]
    random_seed = 42
    finetune = "sound"

@scenario_ingredient.named_config
def vizdoom0():
    scenario = "vizdoom0"
    data_dir = './dataset/vizdoom0'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1
    use_sonic_aim_support = False
    use_auto_aim_support = False

@scenario_ingredient.named_config
def vizdoom1():
    scenario = "vizdoom1"
    data_dir = './dataset/vizdoom1'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1
    use_sonic_aim_support = True
    use_auto_aim_support = False

@scenario_ingredient.named_config
def vizdoom2():
    scenario = "vizdoom2"
    data_dir = './dataset/vizdoom2'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom3():
    scenario = "vizdoom3"
    data_dir = './dataset/vizdoom3'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom4():
    scenario = "vizdoom4"
    data_dir = './dataset/vizdoom4'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom5():
    scenario = "vizdoom5"
    data_dir = './dataset/vizdoom5'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom6():
    scenario = "vizdoom6"
    data_dir = './dataset/vizdoom6'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom7():
    scenario = "vizdoom7"
    data_dir = './dataset/vizdoom7'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom8():
    scenario = "vizdoom8"
    data_dir = './dataset/vizdoom8'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom9():
    scenario = "vizdoom9"
    data_dir = './dataset/vizdoom9'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom10():
    scenario = "vizdoom10"
    data_dir = './dataset/vizdoom10'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom11():
    scenario = "vizdoom11"
    data_dir = './dataset/vizdoom11'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1
@scenario_ingredient.named_config
def vizdoom12():
    scenario = "vizdoom12"
    data_dir = './dataset/vizdoom12'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom13():
    scenario = "vizdoom13"
    data_dir = './dataset/vizdoom13'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom14():
    scenario = "vizdoom14"
    data_dir = './dataset/vizdoom14'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom15():
    scenario = "vizdoom15"
    data_dir = './dataset/vizdoom15'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom16():
    scenario = "vizdoom16"
    data_dir = './dataset/vizdoom16'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom17():
    scenario = "vizdoom17"
    data_dir = './dataset/vizdoom17'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom18():
    scenario = "vizdoom18"
    data_dir = './dataset/vizdoom18'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom19():
    scenario = "vizdoom19"
    data_dir = './dataset/vizdoom19'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1

@scenario_ingredient.named_config
def vizdoom20():
    scenario = "vizdoom20"
    data_dir = './dataset/vizdoom20'
    train_samples = 32000
    test_samples = 8000
    random_seed = 42
    finetune = "sound"
    n_stack = 1