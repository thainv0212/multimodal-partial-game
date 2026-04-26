from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid(
    [
        ("seed", seeds(4)),
    ]
)

_experiment_no_sound = Experiment(
    "doom_basic_no_sound",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --train_dir=train_dir/doom_basic_no_sound",
    _params.generate_params(randomize=False),
)
_experiment_vision_sound_fft = Experiment(
    "doom_basic_vision_fft",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=fft --train_dir=train_dir/doom_basic_vision_fft",
    _params.generate_params(randomize=False),
)
_experiment_vision_sound_raw = Experiment(
    "doom_basic_vision_raw",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=raw --train_dir=train_dir/doom_basic_vision_raw",
    _params.generate_params(randomize=False),
)
_experiment_vision_sound_mel = Experiment(
    "doom_basic_vision_mel",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=mel --train_dir=train_dir/doom_basic_vision_mel",
    _params.generate_params(randomize=False),
)
_experiment_no_vision_sound_mel = Experiment(
    "doom_basic_no_vision_mel",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=mel --encoder_conv_architecture=none --train_dir=train_dir/doom_basic_no_vision_mel",
    _params.generate_params(randomize=False),
)
_experiment_no_vision_sound_fft = Experiment(
    "doom_basic_no_vision_fft",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=fft --encoder_conv_architecture=none --train_dir=train_dir/doom_basic_no_vision_fft",
    _params.generate_params(randomize=False),
)
_experiment_no_vision_sound_raw = Experiment(
    "doom_basic_no_vision_raw",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --train_for_env_steps=2000000 --algo=APPO --env_frameskip=4 --use_rnn=True --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=16 --decorrelate_envs_on_one_worker=False --use_sound --audio_encoder=raw --encoder_conv_architecture=none --train_dir=train_dir/doom_basic_no_vision_raw",
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("doom_basic", experiments=[_experiment_no_sound,
                                                            _experiment_vision_sound_fft,
                                                            _experiment_vision_sound_raw,
                                                            _experiment_vision_sound_mel,
                                                            _experiment_no_vision_sound_mel,
                                                            _experiment_no_vision_sound_fft,
                                                            _experiment_no_vision_sound_raw])

# To run:
# python -m sample_factory.launcher.run --run=sf_examples.vizdoom.experiments.doom_basic --backend=processes --num_gpus=1 --max_parallel=2  --pause_between=0 --experiments_per_gpu=2
