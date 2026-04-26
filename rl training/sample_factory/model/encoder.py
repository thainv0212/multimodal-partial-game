from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.nn import functional as F

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, model_device, nonlinearity
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
import torchaudio

# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(ModelModule):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class MultiInputEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        self.obs_keys = list(sorted(obs_space.keys()))  # always the same order
        self.encoders = nn.ModuleDict()

        out_size = 0

        for obs_key in self.obs_keys:
            shape = obs_space[obs_key].shape

            if len(shape) == 1:
                encoder_fn = MlpEncoder
            elif len(shape) > 1:
                encoder_fn = make_img_encoder
            else:
                raise NotImplementedError(f"Unsupported observation space {obs_space}")

            self.encoders[obs_key] = encoder_fn(cfg, obs_space[obs_key])
            out_size += self.encoders[obs_key].get_out_size()

        self.encoder_out_size = out_size

    def forward(self, obs_dict):
        if len(self.obs_keys) == 1:
            key = self.obs_keys[0]
            return self.encoders[key](obs_dict[key])

        encodings = []
        for key in self.obs_keys:
            x = self.encoders[key](obs_dict[key])
            encodings.append(x)

        return torch.cat(encodings, 1)

    def get_out_size(self) -> int:
        return self.encoder_out_size


class MlpEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        mlp_layers: List[int] = cfg.encoder_mlp_layers
        self.mlp_head = create_mlp(mlp_layers, obs_space.shape[0], nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp_head = torch.jit.script(self.mlp_head)
        self.encoder_out_size = calc_num_elements(self.mlp_head, obs_space.shape)

    def forward(self, obs: Tensor):
        x = self.mlp_head(obs)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class ConvEncoderImpl(nn.Module):
    """
    After we parse all the configuration and figure out the exact architecture of the model,
    we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
    fusion).
    """

    def __init__(self, obs_shape: AttrDict, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module):
        super().__init__()

        conv_layers = []
        for layer in conv_filters:
            if layer == "maxpool_2x2":
                conv_layers.append(nn.MaxPool2d((2, 2)))
            elif isinstance(layer, (list, tuple)):
                inp_ch, out_ch, filter_size, stride = layer
                conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                conv_layers.append(activation)
            else:
                raise NotImplementedError(f"Layer {layer} not supported!")

        self.conv_head = nn.Sequential(*conv_layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)
        self.mlp_layers = create_mlp(extra_mlp_layers, self.conv_head_out_size, activation)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


class ConvEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        input_channels = obs_space.shape[0]
        log.debug(f"{ConvEncoder.__name__}: {input_channels=}")

        if cfg.encoder_conv_architecture == "convnet_simple":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_conv_architecture == "convnet_impala":
            conv_filters = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.encoder_conv_architecture == "convnet_atari":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(f"Unknown encoder architecture {cfg.encoder_conv_architecture}")

        activation = nonlinearity(self.cfg)
        extra_mlp_layers: List[int] = cfg.encoder_conv_mlp_layers
        enc = ConvEncoderImpl(obs_space.shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_space.shape)
        log.debug(f"Conv encoder output size: {self.encoder_out_size}")

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs: Tensor) -> Tensor:
        return self.enc(obs)


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out


class ResnetEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        log.debug("Num input channels: %d", input_ch)

        if cfg.encoder_conv_architecture == "resnet_impala":
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(cfg)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        log.debug(f"Convolutional layer output size: {self.conv_head_out_size}")

        self.mlp_layers = create_mlp(cfg.encoder_conv_mlp_layers, self.conv_head_out_size, activation)

        # should we do torch.jit here?

        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class BaseSoundEncoder(Encoder, ABC):
    def __init__(self, cfg: Config, sampling_rate=22050, fps=35, frame_skip=4):
        super(BaseSoundEncoder, self).__init__(cfg)
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.frame_skip = frame_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left side
        left = x[:, :, 0]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, :, 1]
        right = self.encode_single_channel(right)
        return torch.cat((left, right), dim=1)

    @abstractmethod
    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        pass


class RawEncoder(BaseSoundEncoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace, sampling_rate=22050, fps=35, frame_skip=4):
        super(RawEncoder, self).__init__(cfg)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        assert int(self.num_samples) == self.num_samples

        # Encoder (small 1D conv)
        self.pool = torch.nn.MaxPool1d(2)
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=16, stride=8)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=16, stride=8)
        self.encoder_out_size = calc_num_elements(self, obs_space.shape)

    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        """Shape of x: [batch_size, num_samples]"""
        # Subsample
        x = data[:, ::self.num_to_subsample]

        # Add channel dimension
        x = x[:, None, :]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if x.shape[2] >= 24:
            x = self.conv2(x)
            x = self.pool(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


class FFTEncoder(BaseSoundEncoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace, sampling_rate=22050, fps=35, frame_skip=4):
        super(FFTEncoder, self).__init__(cfg)
        self.num_to_subsample = 8
        self.num_samples = (self.sampling_rate / self.FPS) * self.frame_skip
        self.num_frequencies = self.num_samples / 2
        assert int(self.num_samples) == self.num_samples
        self.num_samples = int(self.num_samples)
        self.num_frequencies = int(self.num_frequencies)

        self.hamming_window = torch.hamming_window(self.num_samples)

        # Subsampler
        self.pool = torch.nn.MaxPool1d(self.num_to_subsample)

        # Encoder (small MLP)
        self.linear1 = torch.nn.Linear(int(self.num_frequencies / self.num_to_subsample), 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.encoder_out_size = calc_num_elements(self, obs_space.shape)

    def _torch_1d_fft_magnitude(self, x: torch.Tensor):
        """Perform 1D FFT on x with shape (batch_size, num_samples), and return magnitudes"""
        # Apply hamming window
        if x.device != self.hamming_window.device:
            self.hamming_window = self.hamming_window.to(x.device)
        x = x * self.hamming_window
        # Add zero imaginery parts
        x = torch.stack((x, torch.zeros_like(x)), dim=-1)
        c = torch.view_as_complex(x)
        ffts = torch.fft.fft(c)
        ffts = torch.view_as_real(ffts)
        # Remove mirrored part
        ffts = ffts[:, :(ffts.shape[1] // 2), :]
        # To magnitudes
        mags = torch.sqrt(ffts[..., 0] ** 2 + ffts[..., 1] ** 2)
        return mags

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        """Shape of x: [batch_size, num_samples]"""
        mags = self._torch_1d_fft_magnitude(data)
        mags = torch.log(mags + 1e-5)

        # Add and remove "channel" dim...
        x = self.pool(mags[:, None, :])[:, 0, :]
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x


class MelSpecEncoder(BaseSoundEncoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace, sampling_rate=22050, fps=35, frame_skip=4):
        super(MelSpecEncoder, self).__init__(cfg)
        self.window_size = int(self.sampling_rate * 0.025)
        self.hop_size = int(self.sampling_rate * 0.01)
        self.n_fft = int(self.sampling_rate * 0.025)
        self.n_mels = 80

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_mels=80,
            n_fft=self.n_fft,
            win_length=self.window_size,
            hop_length=self.hop_size,
            f_min=20,
            f_max=7600,
        )

        # Encoder
        self.conv1 = torch.nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.encoder_out_size = calc_num_elements(self, obs_space.shape)

    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        x = torch.log(self.mel_spectrogram(data) + 1e-5)
        x = torch.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        if x.shape[-1] >= 2:
            x = self.pool(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_img_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Make (most likely convolutional) encoder for image-based observations."""
    if cfg.encoder_conv_architecture.startswith("convnet"):
        return ConvEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("resnet"):
        return ResnetEncoder(cfg, obs_space)
    elif cfg.encoder_conv_architecture.startswith("none"):
        return None
    else:
        raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_architecture}")


def make_sound_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    if cfg.audio_encoder.startswith("raw"):
        return RawEncoder(cfg, obs_space)
    elif cfg.audio_encoder.startswith("fft"):
        return FFTEncoder(cfg, obs_space)
    elif cfg.audio_encoder.startswith("mel"):
        return MelSpecEncoder(cfg, obs_space)
    else:
        raise NotImplementedError(f"Unknown sound encoder architecture {cfg.encoder_conv_architecture}")


def default_make_encoder_func(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """
    Analyze the observation space and create either a convolutional or an MLP encoder depending on
    whether this is an image-based environment or environment with vector observations.
    """
    # we only support dict observation spaces - envs with non-dict obs spaces use a wrapper
    # main subspace used to determine the encoder type is called "obs". For envs with multiple subspaces,
    # this function needs to be overridden (see vizdoom or dmlab encoders for example)
    return MultiInputEncoder(cfg, obs_space)
