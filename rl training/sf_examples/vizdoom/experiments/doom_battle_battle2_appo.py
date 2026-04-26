from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("env", ["doom_battle", "doom_battle2"]),
        ("seed", seeds(4)),
    ]
)

_experiments = [
    Experiment(
        "battle_fs4_no_sound",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_no_sound",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "battle_fs4_sound_raw",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_sound_raw --audio_encoder=raw --use_sound",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "battle_fs4_sound_fft",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_sound_fft --audio_encoder=fft --use_sound",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "battle_fs4_sound_mel",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_sound_mel --audio_encoder=mel --use_sound",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "battle_fs4_no_vision_raw",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_no_vision_raw --audio_encoder=raw --use_sound --encoder_conv_architecture=none",
        _params.generate_params(randomize=False),
    ),
    Experiment(
        "battle_fs4_no_vision_fft",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_no_vision_fft --audio_encoder=fft --use_sound --encoder_conv_architecture=none",
        _params.generate_params(randomize=False),
    ),
Experiment(
        "battle_fs4_no_vision_mel",
        "python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --train_dir=train_dir/battle_fs4_no_vision_mel --audio_encoder=mel --use_sound --encoder_conv_architecture=none",
        _params.generate_params(randomize=False),
    ),
]

RUN_DESCRIPTION = RunDescription("doom_battle_battle2_appo_v1.121.2", experiments=_experiments)
