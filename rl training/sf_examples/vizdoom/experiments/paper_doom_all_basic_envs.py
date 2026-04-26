from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        # ("seed", [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
        ("seed", [0, 1111]),
        # (
        #     "env",
        #     [
        #         "doom_my_way_home",
        #         "doom_deadly_corridor",
        #         "doom_defend_the_center",
        #         "doom_defend_the_line",
        #         "doom_health_gathering",
        #         "doom_health_gathering_supreme",
        #     ],
        # ),
    ]
)
_experiments = [
]
envs = [
    "doom_basic",
    "doom_my_way_home",
    "doom_deadly_corridor",
    "doom_defend_the_center",
    "doom_defend_the_line",
    "doom_health_gathering",
    "doom_health_gathering_supreme",
    "doom_take_cover"
]
for env in envs:
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_no_sound",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --env={env} --train_dir=train_dir/{env}_no_sound",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_vision_raw",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=raw --train_dir=train_dir/{env}_vision_raw",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_vision_fft",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=fft --train_dir=train_dir/{env}_vision_fft",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_vision_mel",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=mel --train_dir=train_dir/{env}_vision_mel",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_no_vision_mel",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=mel --train_dir=train_dir/{env}_no_vision_mel --encoder_conv_architecture=none",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_no_vision_fft",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=fft --train_dir=train_dir/{env}_no_vision_fft --encoder_conv_architecture=none",
            _params.generate_params(randomize=False),
        )
    )
    _experiments.append(
        Experiment(
            f"{env}_basic_envs_fs4_no_vision_fft",
            f"python -m sf_examples.vizdoom.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False  --env={env} --use_sound --audio_encoder=raw --train_dir=train_dir/{env}_no_vision_raw --encoder_conv_architecture=none",
            _params.generate_params(randomize=False),
        )
    )
RUN_DESCRIPTION = RunDescription("paper_doom_basic_envs", experiments=_experiments)
