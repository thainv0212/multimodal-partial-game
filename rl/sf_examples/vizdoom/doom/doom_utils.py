import datetime
import os
from os.path import join
from typing import Optional

from gymnasium.spaces import Discrete
import gymnasium as gym
from gymnasium import ObservationWrapper, RewardWrapper, spaces
import cv2
from sample_factory.envs.env_wrappers import (
    PixelFormatChwWrapper,
    RecordingWrapper,
    ResizeWrapper,
    RewardScalingWrapper,
    TimeLimitWrapper, has_image_observations
)
from sample_factory.utils.utils import debug_log_every_n
from sf_examples.vizdoom.doom.action_space import (
    doom_action_space,
    doom_action_space_basic,
    doom_action_space_discretized_no_weap,
    doom_action_space_extended,
    doom_action_space_full_discretized,
    doom_turn_and_attack_only,
)
from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv
from sf_examples.vizdoom.doom.wrappers.additional_input import DoomAdditionalInput
from sf_examples.vizdoom.doom.wrappers.multiplayer_stats import MultiplayerStatsWrapper
from sf_examples.vizdoom.doom.wrappers.observation_space import SetResolutionWrapper, resolutions
from sf_examples.vizdoom.doom.wrappers.reward_shaping import (
    REWARD_SHAPING_BATTLE,
    REWARD_SHAPING_DEATHMATCH_V0,
    REWARD_SHAPING_DEATHMATCH_V1,
    DoomRewardShapingWrapper,
    true_objective_frags,
    true_objective_winning_the_game,
)
from sf_examples.vizdoom.doom.wrappers.scenario_wrappers.gathering_reward_shaping import DoomGatheringRewardShaping
import numpy as np


class DoomSpec:
    def __init__(
            self,
            name,
            env_spec_file,
            action_space,
            reward_scaling=1.0,
            default_timeout=-1,
            num_agents=1,
            num_bots=0,
            respawn_delay=0,
            timelimit=4.0,
            extra_wrappers=None,
    ):
        self.name = name
        self.env_spec_file = env_spec_file
        self.action_space = action_space
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout

        # 1 for singleplayer, >1 otherwise
        self.num_agents = num_agents

        self.num_bots = num_bots

        self.respawn_delay = respawn_delay
        self.timelimit = timelimit

        # expect list of tuples (wrapper_cls, wrapper_kwargs)
        self.extra_wrappers = extra_wrappers


ADDITIONAL_INPUT = (DoomAdditionalInput, {})  # health, ammo, etc. as input vector
BATTLE_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_BATTLE, true_objective_func=None),
)  # "true" reward None means just the env reward (monster kills)
BOTS_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V0, true_objective_func=true_objective_frags),
)
DEATHMATCH_REWARD_SHAPING = (
    DoomRewardShapingWrapper,
    dict(reward_shaping_scheme=REWARD_SHAPING_DEATHMATCH_V1, true_objective_func=true_objective_winning_the_game),
)

DOOM_ENVS = [
    DoomSpec(
        "doom_basic",
        "basic.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design",
        "basic_new_design.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_2",
        "basic_new_design_2.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_3",
        "basic_new_design_3.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_4",
        "basic_new_design_4.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec("doom_my_way_home", "my_way_home.cfg", doom_action_space_basic(), 1.0),
    DoomSpec("doom_my_way_home_new_design", "my_way_home_new_design.cfg", doom_action_space_basic(), 1.0),
    DoomSpec("doom_deadly_corridor", "deadly_corridor.cfg", doom_action_space_extended(), 0.01),
    DoomSpec("doom_deadly_corridor_new_design", "deadly_corridor_new_design.cfg", doom_action_space_extended(), 0.01),
    DoomSpec("doom_take_cover", "take_cover.cfg", Discrete(1 + 2), 1.0, -1),
    DoomSpec("doom_take_cover_new_design", "take_cover_new_design.cfg", Discrete(1 + 2), 1.0, -1),
    DoomSpec("doom_predict_position", "predict_position.cfg", Discrete(1 + 3), 1.0, 300),
    DoomSpec("doom_predict_position_new_design", "predict_position_new_design.cfg", Discrete(1 + 3), 1.0, 300),
    DoomSpec("doom_defend_the_center", "defend_the_center.cfg", doom_turn_and_attack_only(), 1.0),
    DoomSpec(
        "doom_basic_new_design_5",
        "basic_new_design_5.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_6",
        "basic_new_design_6.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_7",
        "basic_new_design_7.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec(
        "doom_basic_new_design_4_2",
        "basic_new_design_4_2.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec("doom_deathmatch", "deathmatch.cfg", doom_action_space_full_discretized(), 1.0),
    DoomSpec(
        "doom_deathmatch_bots",
        "dwango5_dm_continuous_weap.cfg",
        doom_action_space_full_discretized(),
        1.0,
        int(1e9),
        num_agents=1,
        num_bots=7
    ),
    DoomSpec(
        "doom_basic_new_design_8",
        "basic_new_design_8.cfg",
        Discrete(1 + 3),  # idle, left, right, attack
        reward_scaling=0.01,
        default_timeout=300,
    ),
    DoomSpec( #15
        "doom_my_way_home_A1",
        "my_way_home-A1.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #16
        "doom_my_way_home_A2",
        "my_way_home-A2.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #17
        "doom_my_way_home_A3",
        "my_way_home-A3.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #17
        "doom_my_way_home_A1_b",
        "my_way_home-b-A1.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #18
        "vizdoom18",
        "music_sound.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #19
        "vizdoom19",
        "music_sound_multi.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec( #19
        "vizdoom20",
        "basic_audio.cfg",
        doom_action_space_basic(),
        1.0
    ),

]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise RuntimeError("Unknown Doom env")


# noinspection PyUnusedLocal
def make_doom_env_impl(
        doom_spec,
        cfg=None,
        env_config=None,
        skip_frames=None,
        episode_horizon=None,
        player_id=None,
        num_agents=None,
        max_num_players=None,
        num_bots=0,  # for multi-agent
        custom_resolution=None,
        render_mode: Optional[str] = None,
        **kwargs,
):
    skip_frames = skip_frames if skip_frames is not None else cfg.env_frameskip

    fps = cfg.fps if "fps" in cfg else None
    async_mode = fps == 0

    if player_id is None:
        env = VizdoomEnv(
            doom_spec.action_space,
            doom_spec.env_spec_file,
            skip_frames=skip_frames,
            async_mode=async_mode,
            render_mode=render_mode,
            use_auto_aim_support=cfg['use_auto_aim_support'],
            use_sonic_aim_support=cfg['use_sonic_aim_support'],
        )
    else:
        timelimit = cfg.timelimit if cfg.timelimit is not None else doom_spec.timelimit

        from sf_examples.vizdoom.doom.multiplayer.doom_multiagent import VizdoomEnvMultiplayer

        env = VizdoomEnvMultiplayer(
            doom_spec.action_space,
            doom_spec.env_spec_file,
            player_id=player_id,
            num_agents=num_agents,
            max_num_players=max_num_players,
            num_bots=num_bots,
            skip_frames=skip_frames,
            async_mode=async_mode,
            respawn_delay=doom_spec.respawn_delay,
            timelimit=timelimit,
            render_mode=render_mode,
        )

    record_to = cfg.record_to if "record_to" in cfg else None
    should_record = False
    if env_config is None:
        should_record = True
    elif env_config.worker_index == 0 and env_config.vector_index == 0 and (player_id is None or player_id == 0):
        should_record = True

    if record_to is not None and should_record:
        env = RecordingWrapper(env, record_to, player_id)

    env = MultiplayerStatsWrapper(env)

    # # BotDifficultyWrapper no longer in use
    # if num_bots > 0:
    #     bot_difficulty = cfg.start_bot_difficulty if "start_bot_difficulty" in cfg else None
    #     env = BotDifficultyWrapper(env, bot_difficulty)

    resolution = custom_resolution
    if resolution is None:
        resolution = "256x144" if cfg.wide_aspect_ratio else "160x120"

    assert resolution in resolutions
    env = SetResolutionWrapper(env, resolution)  # default (wide aspect ratio)

    h, w, channels = env.observation_space['img'].shape
    if w != cfg.res_w or h != cfg.res_h:
        env = CustomResizeWrapper(env, cfg.res_w, cfg.res_h, grayscale=False)

    debug_log_every_n(50, "Doom resolution: %s, resize resolution: %r", resolution, (cfg.res_w, cfg.res_h))

    # randomly vary episode duration to somewhat decorrelate the experience
    timeout = doom_spec.default_timeout
    if episode_horizon is not None and episode_horizon > 0:
        timeout = episode_horizon
    if timeout > 0:
        env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=0)

    pixel_format = cfg.pixel_format if "pixel_format" in cfg else "HWC"
    if pixel_format == "CHW":
        env = CustomPixelFormatWrapper(env)

    if doom_spec.extra_wrappers is not None:
        for wrapper_cls, wrapper_kwargs in doom_spec.extra_wrappers:
            env = wrapper_cls(env, **wrapper_kwargs)

    if doom_spec.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_spec.reward_scaling)

    return env


class CustomResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h, grayscale=True, add_channel_dim=False, area_interpolation=False):
        super(CustomResizeWrapper, self).__init__(env)

        self.w = w
        self.h = h
        self.grayscale = grayscale
        self.add_channel_dim = add_channel_dim
        self.interpolation = cv2.INTER_AREA if area_interpolation else cv2.INTER_NEAREST

        if isinstance(env.observation_space, spaces.Dict):
            # TODO: does this even work?
            new_spaces = {}
            for key, space in env.observation_space.spaces.items():
                if key == 'img':
                    new_spaces[key] = self._calc_new_obs_space(space)
                else:
                    new_spaces[key] = space
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = self._calc_new_obs_space(env.observation_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]

        if self.grayscale:
            new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]
        else:
            if len(old_space.shape) > 2:
                channels = old_space.shape[-1]
                new_shape = [self.h, self.w, channels]
            else:
                new_shape = [self.h, self.w, 1] if self.add_channel_dim else [self.h, self.w]

        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    def _convert_obs(self, obs):
        if obs is None:
            return obs

        obs = cv2.resize(obs, (self.w, self.h), interpolation=self.interpolation)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.add_channel_dim:
            return obs[:, :, None]  # add new dimension (expected by tensorflow)
        else:
            return obs

    def _observation(self, obs):
        if isinstance(obs, dict):
            new_obs = {}
            for key, value in obs.items():
                if key == 'img':
                    new_obs[key] = self._convert_obs(value)
                else:
                    new_obs[key] = value
            return new_obs
        else:
            return self._convert_obs(obs)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._observation(obs), reward, terminated, truncated, info


class CustomPixelFormatWrapper(ObservationWrapper):
    """TODO? This can be optimized for VizDoom, can we query CHW directly from VizDoom?"""

    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            img_obs_space = env.observation_space['img']
            self.dict_obs_space = True
        else:
            img_obs_space = env.observation_space
            self.dict_obs_space = False

        if not has_image_observations(img_obs_space):
            raise Exception("Pixel format wrapper only works with image-based envs")

        obs_shape = img_obs_space.shape
        max_num_img_channels = 4

        if len(obs_shape) <= 2:
            raise Exception("Env obs do not have channel dimension?")

        if obs_shape[0] <= max_num_img_channels:
            raise Exception("Env obs already in CHW format?")

        h, w, c = obs_shape
        low, high = img_obs_space.low.flat[0], img_obs_space.high.flat[0]
        new_shape = [c, h, w]

        if self.dict_obs_space:
            dtype = (
                env.observation_space.spaces['img'].dtype
                if env.observation_space.spaces['img'].dtype is not None
                else np.float32
            )
        else:
            dtype = env.observation_space.dtype if env.observation_space.dtype is not None else np.float32

        new_img_obs_space = spaces.Box(low, high, shape=new_shape, dtype=dtype)

        if self.dict_obs_space:
            self.observation_space = env.observation_space
            self.observation_space.spaces['img'] = new_img_obs_space
        else:
            self.observation_space = new_img_obs_space

        self.action_space = env.action_space

    @staticmethod
    def _transpose(obs):
        try:
            return np.transpose(obs, (2, 0, 1))  # HWC to CHW for PyTorch
        except Exception as ex:
            if len(obs.shape) == 2:
                return np.vstack([obs[None], obs[None], obs[None]])
            else:
                raise ex

    def observation(self, observation):
        if observation is None:
            return observation

        if self.dict_obs_space:
            observation['img'] = self._transpose(observation['img'])
        else:
            observation = self._transpose(observation)
        return observation


def make_doom_multiplayer_env(doom_spec, cfg=None, env_config=None, render_mode: Optional[str] = None, **kwargs):
    skip_frames = cfg.env_frameskip

    if cfg.num_bots < 0:
        num_bots = doom_spec.num_bots
    else:
        num_bots = cfg.num_bots

    num_agents = doom_spec.num_agents if cfg.num_agents <= 0 else cfg.num_agents
    max_num_players = num_agents + cfg.num_humans

    is_multiagent = num_agents > 1

    def make_env_func(player_id):
        return make_doom_env_impl(
            doom_spec,
            cfg=cfg,
            player_id=player_id,
            num_agents=num_agents,
            max_num_players=max_num_players,
            num_bots=num_bots,
            skip_frames=1 if is_multiagent else skip_frames,  # multi-agent skipped frames are handled by the wrapper
            env_config=env_config,
            render_mode=render_mode,
            **kwargs,
        )

    if is_multiagent:
        # create a wrapper that treats multiple game instances as a single multi-agent environment

        from sf_examples.vizdoom.doom.multiplayer.doom_multiagent_wrapper import MultiAgentEnv

        env = MultiAgentEnv(
            num_agents=num_agents,
            make_env_func=make_env_func,
            env_config=env_config,
            skip_frames=skip_frames,
            render_mode=render_mode,
        )
    else:
        # if we have only one agent, there's no need for multi-agent wrapper
        from sf_examples.vizdoom.doom.multiplayer.doom_multiagent_wrapper import init_multiplayer_env

        env = init_multiplayer_env(make_env_func, player_id=0, env_config=env_config)

    return env


def make_doom_env(env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    spec = doom_env_by_name(env_name)
    return make_doom_env_from_spec(spec, env_name, cfg, env_config, render_mode, **kwargs)


def make_doom_env_from_spec(spec, _env_name, cfg, env_config, render_mode: Optional[str] = None, **kwargs):
    """
    Makes a Doom environment from a DoomSpec instance.
    _env_name is unused but we keep it, so functools.partial(make_doom_env_from_spec, env_spec) can registered
    in Sample Factory (first argument in make_env_func is expected to be the env_name).
    """

    if "record_to" in cfg and cfg.record_to:
        tstamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        cfg.record_to = join(cfg.record_to, f"{cfg.experiment}", tstamp)
        if not os.path.isdir(cfg.record_to):
            os.makedirs(cfg.record_to)
    else:
        cfg.record_to = None

    if spec.num_agents > 1 or spec.num_bots > 0:
        # requires multiplayer setup (e.g. at least a host, not a singleplayer game)
        return make_doom_multiplayer_env(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
    else:
        return make_doom_env_impl(spec, cfg=cfg, env_config=env_config, render_mode=render_mode, **kwargs)
