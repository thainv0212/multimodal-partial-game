import copy
import os

os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
import random
import re
import time
from os.path import join
from threading import Thread
from typing import Dict, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
from filelock import FileLock, Timeout
from gymnasium.utils import seeding
from gymnasium import ObservationWrapper, spaces
from vizdoom import SamplingRate
from vizdoom.vizdoom import AutomapMode, DoomGame, Mode, ScreenResolution

from sample_factory.algo.utils.spaces.discretized import Discretized
from sample_factory.utils.utils import log, project_tmp_dir
import vizdoom as vzd
import math
from torch import Tensor
from typing import Dict, Optional, Sequence, Union
from gymnasium.spaces import Discrete


def doom_lock_file(max_parallel):
    """
    Doom instances tend to have problems starting when a lot of them are initialized in parallel.
    This is not a problem during normal execution once the envs are initialized.

    The "sweet spot" for the number of envs that can be initialized in parallel is about 5-10.
    Here we use file locking mechanism to ensure that only a limited amount of envs are being initialized at the same
    time.
    This tends to be more of a problem for multiplayer envs.

    This also has an advantage of working across completely independent process groups, e.g. different experiments.
    """
    lock_filename = f"doom_{random.randrange(0, max_parallel):03d}.lockfile"

    tmp_dir = project_tmp_dir()
    lock_path = join(tmp_dir, lock_filename)
    return lock_path


def key_to_action_default(key):
    """
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_RIGHT
    MOVE_LEFT
    SELECT_WEAPON1
    SELECT_WEAPON2
    SELECT_WEAPON3
    SELECT_WEAPON4
    SELECT_WEAPON5
    SELECT_WEAPON6
    SELECT_WEAPON7
    ATTACK
    SPEED
    TURN_LEFT_RIGHT_DELTA
    """
    from pynput.keyboard import Key

    # health gathering
    action_table = {
        Key.left: 0,
        Key.right: 1,
        Key.up: 2,
        Key.down: 3,
    }

    # action_table = {
    #     Key.up: 0,
    #     Key.down: 1,
    #     Key.alt: 6,
    #     Key.ctrl: 11,
    #     Key.shift: 12,
    #     Key.space: 13,
    #     Key.right: 'turn_right',
    #     Key.left: 'turn_left',
    # }

    return action_table.get(key, None)


class VizdoomEnv(gym.Env):
    def __init__(
            self,
            action_space,
            config_file,
            coord_limits=None,
            max_histogram_length=200,
            show_automap=False,
            skip_frames=1,
            async_mode=False,
            record_to=None,
            render_mode: Optional[str] = None,
            use_auto_aim_support: bool = False,
            use_sonic_aim_support: bool = False,
            buddha_mode: bool = False
    ):
        self.initialized = False

        # essential game data
        self.game = None
        self.state = None
        self.curr_seed = 0
        self.rng = None
        self.skip_frames = skip_frames
        self.async_mode = async_mode

        # optional - for topdown view rendering and visitation heatmaps
        self.show_automap = show_automap
        self.coord_limits = coord_limits

        # can be adjusted after the environment is created (but before any reset() call) via observation space wrapper
        self.screen_w, self.screen_h, self.channels = 640, 480, 3
        self.screen_resolution = ScreenResolution.RES_640X480
        self.calc_observation_space()

        self.black_screen = None

        # provided as a part of environment definition, since these depend on the scenario and
        # can be quite complex multi-discrete spaces
        self.action_space = action_space
        self.composite_action_space = hasattr(self.action_space, "spaces")

        self.delta_actions_scaling_factor = 7.5

        if os.path.isabs(config_file):
            self.config_path = config_file
        else:
            scenarios_dir = join(os.path.dirname(__file__), "scenarios")
            self.config_path = join(scenarios_dir, config_file)
            if not os.path.isfile(self.config_path):
                log.warning(
                    "File %s not found in scenarios dir %s. Consider providing absolute path?",
                    config_file,
                    scenarios_dir,
                )

        self.variable_indices = self._parse_variable_indices(self.config_path)

        # only created if we call render() method
        self.screen = None

        # record full episodes using VizDoom recording functionality
        self.record_to = record_to
        self.curr_demo_dir = None

        self.is_multiplayer = False  # overridden in derived classes

        # (optional) histogram to track positional coverage
        # do not pass coord_limits if you don't need this, to avoid extra calculation
        self.max_histogram_length = max_histogram_length
        self.current_histogram, self.previous_histogram = None, None
        if self.coord_limits:
            x = self.coord_limits[2] - self.coord_limits[0]
            y = self.coord_limits[3] - self.coord_limits[1]
            if x > y:
                len_x = self.max_histogram_length
                len_y = int((y / x) * self.max_histogram_length)
            else:
                len_x = int((x / y) * self.max_histogram_length)
                len_y = self.max_histogram_length
            self.current_histogram = np.zeros((len_x, len_y), dtype=np.int32)
            self.previous_histogram = np.zeros_like(self.current_histogram)

        # helpers for human play with pynput keyboard input
        self._terminate = False
        self._current_actions = []
        self._actions_flattened = None

        self._prev_info = None
        self._last_episode_info = None

        self._num_episodes = 0

        self.mode = "algo"

        self.render_mode = render_mode

        self.seed()
        self.use_auto_aim_support = use_auto_aim_support
        self.use_sonic_aim_support = use_sonic_aim_support
        self.last_sonic_time = time.time()
        self.buddha_mode = buddha_mode

    def seed(self, seed: Optional[int] = None):
        """
        Used to seed the actual Doom env.
        If None is passed, the seed is generated randomly.
        """
        self.rng, self.curr_seed = seeding.np_random(seed=seed)
        self.curr_seed = self.curr_seed % (2 ** 32)  # Doom only supports 32-bit seeds
        return [self.curr_seed, self.rng]

    def calc_observation_space(self):
        self.aud_len = 2520
        sound_high = [[32767, 32767]] * self.aud_len
        sound_low = [[-32767, -32767]] * self.aud_len
        self.observation_space_img = gym.spaces.Box(0, 255, (self.screen_h, self.screen_w, self.channels),
                                                    dtype=np.uint8)
        self.observation_space = gym.spaces.Dict({
            'img': self.observation_space_img,
            'sound': gym.spaces.Box(
                low=np.array(sound_low, dtype=np.int16), high=np.array(sound_high, dtype=np.int16),
            ),
        })

    def _set_game_mode(self, mode):
        if mode == "replay":
            self.game.set_mode(Mode.PLAYER)
        else:
            if self.async_mode:
                log.info("Starting in async mode! Use this only for testing, otherwise PLAYER mode is much faster")
                self.game.set_mode(Mode.ASYNC_PLAYER)
            else:
                self.game.set_mode(Mode.PLAYER)

    def _create_doom_game(self, mode):
        self.game = DoomGame()
        self.game.load_config(self.config_path)
        self.game.set_screen_resolution(self.screen_resolution)
        self.game.set_seed(self.curr_seed)
        self.game.set_audio_buffer_enabled(True)
        self.game.set_audio_sampling_rate(SamplingRate.SR_22050)
        self.game.set_audio_buffer_size(self.skip_frames)
        if self.use_auto_aim_support:
            self.game.set_available_buttons(self.game.get_available_buttons() + [vzd.Button.TURN_LEFT_RIGHT_DELTA])
            self.game.set_objects_info_enabled(True)

        # if self.use_auto_aim_support:
        #     self.game.set_console_enabled(True)

        if self.use_sonic_aim_support:
            self.game.add_game_args("-file ./gmc_code/rl/data_modules/envs/vizdoom/sound.wad")
            self.game.set_objects_info_enabled(True)
            # self.game.set_console_enabled(True)

        if mode == "algo":
            self.game.set_window_visible(False)
        elif mode == "human" or mode == "replay":
            self.game.add_game_args("+freelook 1")
            self.game.set_window_visible(True)
        else:
            raise Exception("Unsupported mode")

        self._set_game_mode(mode)
        self.last_sonic_time = time.time()

    def _game_init(self, with_locking=True, max_parallel=10):
        lock_file = lock = None
        if with_locking:
            lock_file = doom_lock_file(max_parallel)
            lock = FileLock(lock_file)

        init_attempt = 0
        while True:
            init_attempt += 1
            try:
                if with_locking:
                    with lock.acquire(timeout=20):
                        self.game.init()
                else:
                    self.game.init()

                break
            except Timeout:
                if with_locking:
                    log.debug(
                        "Another process currently holds the lock %s, attempt: %d",
                        lock_file,
                        init_attempt,
                    )
            except Exception as exc:
                log.warning("VizDoom game.init() threw an exception %r. Terminate process...", exc)
                from sample_factory.envs.env_utils import EnvCriticalError

                raise EnvCriticalError()

    def initialize(self):
        self._create_doom_game(self.mode)

        # (optional) top-down view provided by the game engine
        if self.show_automap:
            self.game.set_automap_buffer_enabled(True)
            self.game.set_automap_mode(AutomapMode.OBJECTS)
            self.game.set_automap_rotate(False)
            self.game.set_automap_render_textures(False)

            # self.game.add_game_args("+am_restorecolors")
            # self.game.add_game_args("+am_followplayer 1")
            background_color = "ffffff"
            self.game.add_game_args("+viz_am_center 1")
            self.game.add_game_args("+am_backcolor " + background_color)
            self.game.add_game_args("+am_tswallcolor dddddd")
            # self.game.add_game_args("+am_showthingsprites 0")
            self.game.add_game_args("+am_yourcolor " + background_color)
            self.game.add_game_args("+am_cheat 0")
            self.game.add_game_args("+am_thingcolor 0000ff")  # player color
            self.game.add_game_args("+am_thingcolor_item 00ff00")
            # self.game.add_game_args("+am_thingcolor_citem 00ff00")

        self._game_init()
        self.initialized = True
        self.game.set_sound_enabled(True)

        # print("=" * 60)
        # print("VIZDOOM AUDIO CONFIGURATION CHECK:")
        # print(f"Sound enabled: {self.game.is_sound_enabled()}")
        # print(f"Audio buffer enabled: {self.game.is_audio_buffer_enabled()}")
        # print(f"Audio sampling rate: {self.game.get_audio_sampling_rate()}")
        # print(f"Audio buffer size: {self.game.get_audio_buffer_size()}")
        # print(f"Available buttons: {self.game.get_available_buttons()}")
        # print("=" * 60)

    def _ensure_initialized(self):
        if not self.initialized:
            self.initialize()

    @staticmethod
    def _parse_variable_indices(config):
        with open(config, "r") as config_file:
            lines = config_file.readlines()
        lines = [ln.strip() for ln in lines]

        variable_indices = {}

        for line in lines:
            if line.startswith("#"):
                continue  # comment

            variables_syntax = r"available_game_variables[\s]*=[\s]*\{(.*)\}"
            match = re.match(variables_syntax, line)
            if match is not None:
                variables_str = match.groups()[0]
                variables_str = variables_str.strip()
                variables = variables_str.split(" ")
                for i, variable in enumerate(variables):
                    variable_indices[variable] = i
                break

        return variable_indices

    def _black_screen(self):
        if self.black_screen is None:
            self.black_screen = np.zeros(self.observation_space['img'].shape, dtype=np.uint8)
        return self.black_screen

    def _game_variables_dict(self, state):
        game_variables = state.game_variables
        variables = {}
        if game_variables is not None:
            for variable, idx in self.variable_indices.items():
                variables[variable] = game_variables[idx]
        return variables

    @staticmethod
    def demo_path(episode_idx, record_to):
        demo_name = f"e{episode_idx:03d}.lmp"
        demo_path_ = join(record_to, demo_name)
        demo_path_ = os.path.normpath(demo_path_)
        return demo_path_

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        if "seed" in kwargs:
            self.seed(kwargs["seed"])

        self._ensure_initialized()

        episode_started = False
        if self.record_to and not self.is_multiplayer:
            # does not work in multiplayer (uses different mechanism)
            if not os.path.exists(self.record_to):
                os.makedirs(self.record_to)

            demo_path = self.demo_path(self._num_episodes, self.record_to)
            self.curr_demo_dir = os.path.dirname(demo_path)
            log.warning(f"Recording episode demo to {demo_path=}")

            if len(demo_path) > 101:
                log.error(f"Demo path {len(demo_path)=}>101, will not record demo")
                log.error(
                    "This seems to be a bug in VizDoom, please just use a shorter demo path, i.e. set --record_to to /tmp/doom_recs"
                )
            else:
                self.game.new_episode(demo_path)
                self.game.send_game_command("buddha")
                episode_started = True

        if self._num_episodes > 0 and not episode_started:
            # no demo recording (default)
            self.game.new_episode()
            self.game.send_game_command("buddha")

        self.state = self.game.get_state()
        img = None
        try:
            img = self.state.screen_buffer
            audio = self.state.audio_buffer
        except AttributeError:
            # sometimes Doom does not return screen buffer at all??? Rare bug
            pass

        if img is None:
            log.error("Game returned None screen buffer! This is not supposed to happen!")
            img = self._black_screen()
            audio = self.zeros([2520, 0])

        # Swap current and previous histogram
        if self.current_histogram is not None and self.previous_histogram is not None:
            swap = self.current_histogram
            self.current_histogram = self.previous_histogram
            self.previous_histogram = swap
            self.current_histogram.fill(0)

        self._actions_flattened = None
        self._last_episode_info = copy.deepcopy(self._prev_info)
        self._prev_info = None

        self._num_episodes += 1

        return {'img': np.transpose(img, (1, 2, 0)),
                'audio': audio}, {}  # since Gym 0.26.0, we return dict as second return value

    def _convert_actions(self, actions):
        """Convert actions from gym action space to the action space expected by Doom game."""

        if self.composite_action_space:
            # composite action space with multiple subspaces
            spaces = self.action_space.spaces
        else:
            # simple action space, e.g. Discrete. We still treat it like composite of length 1
            spaces = (self.action_space,)
            actions = (actions,)

        actions_flattened = []
        for i, action in enumerate(actions):
            if isinstance(spaces[i], Discretized):
                # discretized continuous action
                # check discretized first because it's a subclass of gym.spaces.Discrete
                # the order of if clauses here matters! DON'T CHANGE THE ORDER OF IFS!

                continuous_action = spaces[i].to_continuous(action)
                actions_flattened.append(continuous_action)
            elif isinstance(spaces[i], gym.spaces.Discrete):
                # standard discrete action
                num_non_idle_actions = spaces[i].n - 1
                action_one_hot = np.zeros(num_non_idle_actions, dtype=np.uint8)
                if action > 0:
                    action_one_hot[action - 1] = 1  # 0th action in each subspace is a no-op

                actions_flattened.extend(action_one_hot)
            elif isinstance(spaces[i], gym.spaces.Box):
                # continuous action
                actions_flattened.extend(list(action * self.delta_actions_scaling_factor))
            else:
                raise NotImplementedError(f"Action subspace type {type(spaces[i])} is not supported!")

        return actions_flattened

    def _vizdoom_variables_bug_workaround(self, info, done):
        """Some variables don't get reset to zero on game.new_episode(). This fixes it (also check overflow?)."""
        if done and "DAMAGECOUNT" in info:
            log.info("DAMAGECOUNT value on done: %r", info.get("DAMAGECOUNT"))

        if self._last_episode_info is not None:
            bugged_vars = ["DEATHCOUNT", "HITCOUNT", "DAMAGECOUNT"]
            for v in bugged_vars:
                if v in info:
                    info[v] -= self._last_episode_info.get(v, 0)

    def _process_game_step(self, state, done, info):
        if not done:
            observation = np.transpose(state.screen_buffer, (1, 2, 0))
            audio_buffer = state.audio_buffer
            game_variables = self._game_variables_dict(state)
            info.update(self.get_info(game_variables))
            self._update_histogram(info)
            self._prev_info = copy.copy(info)
        else:
            observation = self._black_screen()
            audio_buffer = np.zeros([2520, 0])

            # when done=True Doom does not allow us to call get_info, so we provide info from the last frame
            # print("checking info:", info is None, self._prev_info is None)
            if info is None:
                info = {}
            if self._prev_info is not None:
                info.update(self._prev_info)

        self._vizdoom_variables_bug_workaround(info, done)

        return {'img': observation, 'audio': audio_buffer}, done, info

    def step(self, actions) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        if self._actions_flattened is not None:
            # provided externally, e.g. via human play
            actions_flattened = self._actions_flattened
            self._actions_flattened = None
        else:
            actions_flattened = self._convert_actions(actions)

        if self.use_auto_aim_support:
            turn_left, turn_right = auto_aim(-1, 45, self.game.get_state().objects, P_Name="DoomPlayer")
            turn_delta = (turn_right - turn_left) * 3
            actions_flattened += [turn_delta]
        if self.use_sonic_aim_support:
            self.last_sonic_time = sonic_aim(500, 45, self.game.get_state().objects, self.game, self.last_sonic_time)
        default_info = {"num_frames": self.skip_frames}
        reward = self.game.make_action(actions_flattened, self.skip_frames)
        state = self.game.get_state()
        done = self.game.is_episode_finished()

        observation, done, info = self._process_game_step(state, done, default_info)
        # if not done and 'audio' in observation:
        #         audio = observation['audio']
        #         non_zero = np.count_nonzero(audio)
        #         if non_zero > 0:
        #             print(f"[STEP] Audio has {non_zero} non-zero samples! min={audio.min()}, max={audio.max()}")
        # Gym 0.26.0 changes
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        mode = self.render_mode
        if mode is None:
            return

        try:
            img = self.game.get_state().screen_buffer
            img = np.transpose(img, [1, 2, 0])
            if mode == "rgb_array":
                return img

            h, w = img.shape[:2]
            render_h, render_w = h, w
            max_w = 1280
            if w < max_w:
                render_w = max_w
                render_h = int(max_w * h / w)
                img = cv2.resize(img, (render_w, render_h))

            import pygame

            if self.screen is None:
                pygame.init()
                pygame.display.init()
                self.screen = pygame.display.set_mode((render_w, render_h))

            pygame.surfarray.blit_array(self.screen, img.swapaxes(0, 1))
            pygame.display.update()

            return img
        except AttributeError:
            return None

    def close(self):
        try:
            if self.game is not None:
                self.game.close()
        except RuntimeError as exc:
            log.warning("Runtime error in VizDoom game close(): %r", exc)

        # if self.viewer is not None:
        #     self.viewer.close()
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

    def get_info(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())

        info_dict = {"pos": self.get_positions(variables)}
        info_dict.update(variables)
        return info_dict

    def get_info_all(self, variables=None):
        if variables is None:
            variables = self._game_variables_dict(self.game.get_state())
        info = self.get_info(variables)
        if self.previous_histogram is not None:
            info["previous_histogram"] = self.previous_histogram
        return info

    def get_positions(self, variables):
        return self._get_positions(variables)

    @staticmethod
    def _get_positions(variables):
        have_coord_data = True
        required_vars = ["POSITION_X", "POSITION_Y", "ANGLE"]
        for required_var in required_vars:
            if required_var not in variables:
                have_coord_data = False
                break

        x = y = a = np.nan
        if have_coord_data:
            x = variables["POSITION_X"]
            y = variables["POSITION_Y"]
            a = variables["ANGLE"]

        return {"agent_x": x, "agent_y": y, "agent_a": a}

    def get_automap_buffer(self):
        if self.game.is_episode_finished():
            return None
        state = self.game.get_state()
        map_ = state.automap_buffer
        map_ = np.swapaxes(map_, 0, 2)
        map_ = np.swapaxes(map_, 0, 1)
        return map_

    def _update_histogram(self, info, eps=1e-8):
        if self.current_histogram is None:
            return
        agent_x, agent_y = info["pos"]["agent_x"], info["pos"]["agent_y"]

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - self.coord_limits[0]) / (self.coord_limits[2] - self.coord_limits[0])
        dy = (agent_y - self.coord_limits[1]) / (self.coord_limits[3] - self.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - eps) * self.current_histogram.shape[0])
        dy = int((dy - eps) * self.current_histogram.shape[1])

        self.current_histogram[dx, dy] += 1

    def _key_to_action(self, key):
        if hasattr(self.action_space, "key_to_action"):
            return self.action_space.key_to_action(key)
        else:
            return key_to_action_default(key)

    def _keyboard_on_press(self, key):
        from pynput.keyboard import Key

        if key == Key.esc:
            self._terminate = True
            return False

        action = self._key_to_action(key)
        if action is not None:
            if action not in self._current_actions:
                self._current_actions.append(action)

    def _keyboard_on_release(self, key):
        action = self._key_to_action(key)
        if action is not None:
            if action in self._current_actions:
                self._current_actions.remove(action)

    # noinspection PyProtectedMember
    @staticmethod
    def play_human_mode(env, skip_frames=1, num_episodes=3, num_actions=None):
        from pynput.keyboard import Listener

        doom = env.unwrapped
        doom.skip_frames = 1  # handled by this script separately

        # noinspection PyProtectedMember
        def start_listener():
            with Listener(on_press=doom._keyboard_on_press, on_release=doom._keyboard_on_release) as listener:
                listener.join()

        listener_thread = Thread(target=start_listener)
        listener_thread.start()

        for episode in range(num_episodes):
            doom.mode = "human"
            env.reset()
            last_render_time = time.time()
            time_between_frames = 1.0 / 35.0

            total_rew = 0.0

            while not doom.game.is_episode_finished() and not doom._terminate:
                num_actions = 14 if num_actions is None else num_actions
                turn_delta_action_idx = num_actions - 1

                actions = [0] * num_actions
                for action in doom._current_actions:
                    if isinstance(action, int):
                        actions[action] = 1  # 1 for buttons currently pressed, 0 otherwise
                    else:
                        if action == "turn_left":
                            actions[turn_delta_action_idx] = -doom.delta_actions_scaling_factor
                        elif action == "turn_right":
                            actions[turn_delta_action_idx] = doom.delta_actions_scaling_factor

                for frame in range(skip_frames):
                    doom._actions_flattened = actions
                    _, rew, _, _, _ = env.step(actions)

                    new_total_rew = total_rew + rew
                    if new_total_rew != total_rew:
                        log.info("Reward: %.3f, total: %.3f", rew, new_total_rew)
                    total_rew = new_total_rew
                    state = doom.game.get_state()

                    verbose = True
                    if state is not None and verbose:
                        info = doom.get_info()
                        print(
                            "Health:",
                            info["HEALTH"],
                            # 'Weapon:', info['SELECTED_WEAPON'],
                            # 'ready:', info['ATTACK_READY'],
                            # 'ammo:', info['SELECTED_WEAPON_AMMO'],
                            # 'pc:', info['PLAYER_COUNT'],
                            # 'dmg:', info['DAMAGECOUNT'],
                        )

                    time_since_last_render = time.time() - last_render_time
                    time_wait = time_between_frames - time_since_last_render

                    if doom.show_automap and state.automap_buffer is not None:
                        map_ = state.automap_buffer
                        map_ = np.swapaxes(map_, 0, 2)
                        map_ = np.swapaxes(map_, 0, 1)
                        cv2.imshow("ViZDoom Automap Buffer", map_)
                        if time_wait > 0:
                            cv2.waitKey(int(time_wait) * 1000)
                    else:
                        if time_wait > 0:
                            time.sleep(time_wait)

                    last_render_time = time.time()

            if doom.show_automap:
                cv2.destroyAllWindows()

        log.debug("Press ESC to exit...")
        listener_thread.join()

    # noinspection PyProtectedMember
    @staticmethod
    def replay(env, rec_path):
        doom = env.unwrapped
        doom.mode = "replay"
        doom._ensure_initialized()
        doom.game.replay_episode(rec_path)

        episode_reward = 0
        start = time.time()

        while not doom.game.is_episode_finished():
            doom.game.advance_action()
            r = doom.game.get_last_reward()
            episode_reward += r
            log.info("Episode reward: %.3f, time so far: %.1f s", episode_reward, time.time() - start)

        log.info("Finishing replay")
        doom.close()


# Enemy health values (Doom II standard + Others found in ViZDoom)
ENEMY_HEALTH = {
    "Zombieman": 20, "ShotgunGuy": 30, "ChaingunGuy": 70, "MarineChainsawVzd": 70, "DoomImp": 60,
    "Demon": 150, "Spectre": 150, "Cacodemon": 400, "HellKnight": 500,
    "BaronOfHell": 1000, "Arachnotron": 500, "Revenant": 300, "Fatso": 600,
    "PainElemental": 400, "Archvile": 700, "SpiderMastermind": 3000,
    "Cyberdemon": 4000, "WolfensteinSS": 50, "LostSoul": 100
}


def auto_aim(Distance_T, Angle_T, objects, P_Name="DoomPlayer"):
    """
    Auto-aim system for ViZDoom that prioritizes closer enemies and those with higher base HP.

    Parameters:
    - Distance_T: Max range for auto-aim (-1 for unlimited).
    - Angle_T: Max angl-e for auto-aim (-1 for unlimited).
    - objects: List of objects from the game state.
    - player_names: List of potential player names (default includes AI and human players).

    Returns:
    - TurnLeft: Speed of turning left (0 if not turning left)
    - TurnRight: Speed of turning right (0 if not turning right)
    """

    # Initialize variables
    Player_Coordination = None
    Player_Angle = None
    Target_Enemy = None
    Target_Distance = float("inf")
    Target_Angle = None
    Target_HP = -1  # Start with lowest possible HP so stronger enemies are prioritized

    # Locate player
    for obj in objects:
        if obj.name == P_Name:
            Player_Coordination = [obj.position_x, obj.position_y]
            Player_Angle = obj.angle
            break
    else:
        # print("[DEBUG] ❌ Player not found! Aim assist disabled.")
        return 0, 0  # No action taken

    # Process enemies and find the best one to aim at
    for obj in objects:
        if obj.name in ENEMY_HEALTH:
            enemy_pos = [obj.position_x, obj.position_y]

            # Calculate distance from player
            distance = math.sqrt(
                (enemy_pos[0] - Player_Coordination[0]) ** 2 +
                (enemy_pos[1] - Player_Coordination[1]) ** 2
            )

            # Calculate angle difference (in degrees)
            angle_to_enemy = math.degrees(math.atan2(
                enemy_pos[1] - Player_Coordination[1],
                enemy_pos[0] - Player_Coordination[0]
            ))
            angle_diff = angle_to_enemy - Player_Angle

            # Normalize angle difference (-180 to 180 degrees)
            angle_diff = (angle_diff + 180) % 360 - 180

            # Debugging Output
            # print(f"[DEBUG] 🔍 Checking enemy {obj.name} at distance {distance:.2f}, angle diff {angle_diff:.2f}")

            # Ensure enemy detection is within allowed range
            if (Distance_T == -1 or distance <= Distance_T) and \
                    (Angle_T == -1 or abs(angle_diff) <= Angle_T):

                # Get enemy base HP
                enemy_hp = ENEMY_HEALTH[obj.name]

                # Prioritize closest first, then highest HP if distances are equal
                if distance < Target_Distance or (distance == Target_Distance and enemy_hp > Target_HP):
                    Target_Enemy = obj
                    Target_Distance = distance
                    Target_Angle = angle_diff
                    Target_HP = enemy_hp

    if Target_Enemy is None:
        # print("[DEBUG] ❌ No valid targets in range.")
        return 0, 0  # No action taken

    # print(f"[DEBUG] 🎯 Auto-aiming at: {Target_Enemy.name} | Distance: {Target_Distance:.2f} | HP: {Target_HP} | Angle Diff: {Target_Angle:.2f}")

    # Adjust turn speed dynamically
    turn_speed = min(abs(Target_Angle) * 0.1, 1)

    # Introduce dead zone to prevent jitter
    if abs(Target_Angle) < 1:  # If already almost aligned, don't turn
        turn_speed = 0
    else:
        turn_speed = min(abs(Target_Angle) * 0.1, 1)  # Normal speed scaling

    # Normalize angle difference (-180 to 180 degrees)
    Target_Angle = (Target_Angle + 180) % 360 - 180
    # print(f"[DEBUG] Angle Diff: {Target_Angle:.2f}")
    # Adjust turn direction based on angle
    if Target_Angle < 0:
        return 0, turn_speed  # Turn Right
    elif Target_Angle > 0:
        return turn_speed, 0  # Turn Left
    else:
        return 0, 0  # No movement (inside dead zone)


def sonic_aim(Distance_T, Angle_T, objects, game, last_play_time, P_Name="DoomPlayer"):
    """
    Sonic-based enemy awareness system for ViZDoom.
    Plays a pre-recorded sound that increases in volume as the enemy is directly in front of the player.

    Parameters:
    - Distance_T: Max range for sound trigger (-1 for unlimited).
    - Angle_T: Max angle for enemy detection (-1 for unlimited).
    - objects: List of objects from the game state.
    - game: The ViZDoom game instance to send sound commands.
    - P_Name: Player's name (default is "DoomPlayer").

    Returns:
    - None (only plays a sound)
    """

    # global last_play_time  # Track last sound play time
    sound_duration = 0.2  # Adjust based on actual length of "saim" sound
    # Initialize variables
    Target_Enemy = None
    Target_Distance = float("inf")
    Target_Angle = None
    Target_HP = -1

    # Locate player
    for obj in objects:
        if obj.name == P_Name:
            Player_Coordination = [obj.position_x, obj.position_y]
            Player_Angle = obj.angle
            break
    else:
        return last_play_time  # No sound played

    # Find the closest enemy within Angle_T
    for obj in objects:
        if obj.name in ENEMY_HEALTH:
            enemy_pos = [obj.position_x, obj.position_y]

            # Calculate distance from player
            distance = math.sqrt(
                (enemy_pos[0] - Player_Coordination[0]) ** 2 +
                (enemy_pos[1] - Player_Coordination[1]) ** 2
            )

            # Calculate angle difference (in degrees)
            angle_to_enemy = math.degrees(math.atan2(
                enemy_pos[1] - Player_Coordination[1],
                enemy_pos[0] - Player_Coordination[0]
            ))
            angle_diff = angle_to_enemy - Player_Angle

            # Normalize angle difference (-180 to 180 degrees)
            angle_diff = (angle_diff + 180) % 360 - 180

            # Ensure enemy detection is within allowed range
            if (Distance_T == -1 or distance <= Distance_T) and \
                    (Angle_T == -1 or abs(angle_diff) <= Angle_T):

                # Get enemy base HP
                enemy_hp = ENEMY_HEALTH[obj.name]

                # Prioritize closest first, then highest HP if distances are equal
                if distance < Target_Distance or (distance == Target_Distance and enemy_hp > Target_HP):
                    Target_Enemy = obj
                    Target_Distance = distance
                    Target_Angle = angle_diff
                    Target_HP = enemy_hp

    if Target_Enemy is None:
        return last_play_time  # No sound played

    # 🔊 Play sound only if enemy is within `Angle_T`
    current_time = time.time()

    if abs(Target_Angle) <= Angle_T and (current_time - last_play_time >= sound_duration):
        # Choose the closest sound file based on angle alignment
        volume_index = round((1.0 - (abs(Target_Angle) / Angle_T)) * 10)  # Scale to 0-10
        volume_index = max(0, min(10, volume_index))  # Ensure it stays within 0-10

        # Play the appropriate sound file
        game.send_game_command(f"playsound saim_{volume_index}")
    return current_time


resolutions = [
    "160x120",
    "200x125",
    "200x150",
    "256x144",
    "256x160",
    "256x192",
    "320x180",
    "320x200",
    "320x240",
    "320x256",
    "400x225",
    "400x250",
    "400x300",
    "512x288",
    "512x320",
    "512x384",
    "640x360",
    "640x400",
    "640x480",
    "800x450",
    "800x500",
    "800x600",
    "1024x576",
    "1024x640",
    "1024x768",
    "1280x720",
    "1280x800",
    "1280x960",
    "1280x1024",
    "1400x787",
    "1400x875",
    "1400x1050",
    "1600x900",
    "1600x1000",
    "1600x1200",
    "1920x1080",
]


class SetResolutionWrapper(gym.Wrapper):
    """Doom wrapper to change screen resolution."""

    def __init__(self, env, target_resolution):
        super(SetResolutionWrapper, self).__init__(env)
        if target_resolution not in resolutions:
            raise gym.error.Error(
                'Error - The specified resolution "{}" is not supported by Vizdoom.'.format(target_resolution),
            )

        orig_obs_space = self.observation_space

        parts = target_resolution.lower().split("x")
        width = int(parts[0])
        height = int(parts[1])
        screen_res = __import__("vizdoom")
        screen_res = getattr(screen_res, "ScreenResolution")
        screen_res = getattr(screen_res, "RES_{}X{}".format(width, height))

        self.unwrapped.screen_w = width
        self.unwrapped.screen_h = height
        self.unwrapped.screen_resolution = screen_res
        self.unwrapped.calc_observation_space()

        if isinstance(orig_obs_space, gym.spaces.Dict):
            new_obs_space = {}
            for key, value in orig_obs_space.spaces.items():
                new_obs_space[key] = self.unwrapped.observation_space[key]
            new_obs_space = gym.spaces.Dict(new_obs_space)
        else:
            new_obs_space = self.unwrapped.observation_space

        self.observation_space = self.unwrapped.observation_space = new_obs_space


class TimeLimitWrapper(gym.core.Wrapper):
    def __init__(self, env, limit, random_variation_steps=0):
        super(TimeLimitWrapper, self).__init__(env)
        self._limit = limit
        self._variation_steps = random_variation_steps
        self._num_steps = 0
        self._terminate_in = self._random_limit()

    def _random_limit(self):
        return np.random.randint(-self._variation_steps, self._variation_steps + 1) + self._limit

    def reset(self, **kwargs):
        self._num_steps = 0
        self._terminate_in = self._random_limit()
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if observation is None:
            return observation, reward, terminated, truncated, info

        self._num_steps += num_env_steps([info])
        if terminated or truncated:
            pass
        elif self._num_steps >= self._terminate_in:
            truncated = True

        return observation, reward, terminated, truncated, info


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


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


def num_env_steps(infos):
    """Calculate number of environment frames in a batch of experience."""

    total_num_frames = 0
    for info in infos:
        total_num_frames += info.get("num_frames", 1)
    return total_num_frames


def make_dones(terminated, truncated):
    """
    Make dones from terminated/truncated (gym 0.26.0 changes).
    Assumes that terminated and truncated are the same type and shape.
    """
    if isinstance(terminated, (bool, np.ndarray, Tensor)):
        return terminated | truncated
    elif isinstance(terminated, Sequence):
        return [t | truncated[i] for i, t in enumerate(terminated)]

    raise ValueError(f"Unknown {type(terminated)=}")


class MultiplayerStatsWrapper(gym.Wrapper):
    """Add to info things like place in the match, gap to leader, kill-death ratio etc."""

    def __init__(self, env):
        super().__init__(env)
        self.timestep = 0
        self.prev_extra_info = dict()

    def _parse_info(self, info, done):
        if (self.timestep % 20 == 0 or done) and "FRAGCOUNT" in info:
            # no need to update these stats every frame
            kdr = info.get("FRAGCOUNT", 0.0) / (info.get("DEATHCOUNT", 0.0) + 1)
            extra_info = {"KDR": float(kdr)}

            player_count = int(info.get("PLAYER_COUNT", 1))
            player_num = int(info.get("PLAYER_NUMBER", 0))
            fragcounts = [int(info.get(f"PLAYER{pi}_FRAGCOUNT", -100000)) for pi in range(1, player_count + 1)]
            places = list(np.argsort(fragcounts))

            final_place = places.index(player_num)
            final_place = player_count - final_place  # inverse, because fragcount is sorted in increasing order
            extra_info["FINAL_PLACE"] = final_place

            if final_place > 1:
                extra_info["LEADER_GAP"] = max(fragcounts) - fragcounts[player_num]
            elif player_count > 1:
                # we won, let's log gap to 2nd place
                assert places.index(player_num) == player_count - 1
                fragcounts.sort(reverse=True)
                extra_info["LEADER_GAP"] = fragcounts[1] - fragcounts[0]  # should be negative or 0
                assert extra_info["LEADER_GAP"] <= 0
            else:
                extra_info["LEADER_GAP"] = 0

            self.prev_extra_info = extra_info
        else:
            extra_info = self.prev_extra_info

        info.update(extra_info)
        return info

    def reset(self, **kwargs):
        self.timestep = 0
        self.prev_extra_info = dict()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if obs is None:
            return obs, reward, terminated, truncated, info

        done = make_dones(terminated, truncated)
        info = self._parse_info(info, done)
        self.timestep += 1
        return obs, reward, terminated, truncated, info


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


def key_to_action_basic(key):
    from pynput.keyboard import Key

    table = {Key.left: 0, Key.right: 1, Key.up: 2, Key.down: 3}
    return table.get(key, None)


def doom_action_space_extended():
    """
    This function assumes the following list of available buttons:
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_LEFT
    MOVE_RIGHT
    ATTACK
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),  # noop, turn left, turn right
            Discrete(3),  # noop, forward, backward
            Discrete(3),  # noop, strafe left, strafe right
            Discrete(2),  # noop, attack
        )
    )

    return space


def doom_action_space_basic():
    """
    TURN_LEFT
    TURN_RIGHT
    MOVE_FORWARD
    MOVE_BACKWARD
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(3),
        )
    )  # noop, turn left, turn right  # noop, forward, backward

    space.key_to_action = key_to_action_basic
    return space


def doom_turn_and_attack_only():
    """
    TURN_LEFT
    TURN_RIGHT
    ATTACK
    """
    space = gym.spaces.Tuple(
        (
            Discrete(3),
            Discrete(2),
        )
    )  # noop, turn left, turn right  # noop, attack

    return space


def doom_action_space_full_discretized(with_use=False):
    """
    MOVE_FORWARD
    MOVE_BACKWARD
    MOVE_RIGHT
    MOVE_LEFT
    SELECT_WEAPON1
    SELECT_WEAPON2
    SELECT_WEAPON3
    SELECT_WEAPON4
    SELECT_WEAPON5
    SELECT_WEAPON6
    SELECT_WEAPON7
    ATTACK
    SPEED
    TURN_LEFT_RIGHT_DELTA
    """
    spaces = [
        Discrete(3),  # noop, forward, backward
        Discrete(3),  # noop, move right, move left
        Discrete(8),  # noop, select weapons 1-7
        Discrete(2),  # noop, attack
        Discrete(2),  # noop, sprint
    ]
    if with_use:
        spaces.append(Discrete(2))  # noop, use

    spaces.append(Discretized(21, min_action=-12.5, max_action=12.5))  # turning using discretized continuous control

    return gym.spaces.Tuple(spaces)


DOOM_ENVS = [
    DoomSpec( #5
        "doom_my_way_home_enhanced",
        "my_way_home_new_design.cfg",
        doom_action_space_basic(),
        1.0
    ),
    DoomSpec("doom_defend_the_center", "defend_the_center.cfg", doom_turn_and_attack_only(), 1.0),
    #8
    DoomSpec("doom_deathmatch", "deathmatch.cfg", doom_action_space_full_discretized(), 1.0),
    DoomSpec( #18
        "music_find_single",
        "music_sound.cfg",
        doom_action_space_basic(),
        1.0
    )
]


def make_vizdoom_env(doom_spec, use_auto_aim_support=False, use_sonic_aim_support=False, res_w=128, res_h=72,
                     skip_frames=4, async_mode=False, render_mode='human', buddha_mode=True):
    env = VizdoomEnv(
        doom_spec.action_space,
        doom_spec.env_spec_file,
        skip_frames=skip_frames,
        async_mode=async_mode,
        render_mode=render_mode,
        use_auto_aim_support=use_auto_aim_support,
        use_sonic_aim_support=use_sonic_aim_support,
        buddha_mode=buddha_mode,
    )
    env = MultiplayerStatsWrapper(env)
    resolution = "160x120"
    env = SetResolutionWrapper(env, resolution)
    h, w, channels = env.observation_space['img'].shape
    if w != res_w or h != res_h:
        env = CustomResizeWrapper(env, res_w, res_h, grayscale=False)
    env = TimeLimitWrapper(env, limit=300, random_variation_steps=0)
    env = CustomPixelFormatWrapper(env)
    return env


if __name__ == "__main__":
    env = make_vizdoom_env(DOOM_ENVS[0], False, False, 160, 120)
    count = 0
    temp = env.reset()
    while count < 300:
        action = env.action_space.sample()
        env.render()
        obs, rew, terminated, truncated, infos = env.step(action)
        count += 1
        if terminated or truncated:
            env.reset()
            continue
        print(count)
    env.close()
    print('a')
