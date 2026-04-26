from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from abc import ABC, abstractmethod


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class VizdoomEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32


class VizdoomConvEncoder(VizdoomEncoder):
    def __init__(self, obs_shape=(3, 72, 128), conv_architecture="convnet_simple", mpl_layers=[512], non_linear="elu"):
        super().__init__()

        input_channels = obs_shape[0]
        print(f"{VizdoomConvEncoder.__name__}: {input_channels=}")

        if conv_architecture == "convnet_simple":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif conv_architecture == "convnet_impala":
            conv_filters = [[input_channels, 16, 8, 4], [16, 32, 4, 2]]
        elif conv_architecture == "convnet_atari":
            conv_filters = [[input_channels, 32, 8, 4], [32, 64, 4, 2], [64, 64, 3, 1]]
        else:
            raise NotImplementedError(f"Unknown encoder architecture {conv_architecture}")

        activation = nonlinearity(non_linear)
        extra_mlp_layers: List[int] = mpl_layers
        enc = ConvEncoderImpl(obs_shape, conv_filters, extra_mlp_layers, activation)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape)
        print(f"Conv encoder output size: {self.encoder_out_size}")

    def get_out_size(self) -> int:
        return self.encoder_out_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.enc(obs)


class ConvEncoderImpl(nn.Module):
    """
    After we parse all the configuration and figure out the exact architecture of the model,
    we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
    fusion).
    """

    def __init__(self, obs_shape, conv_filters: List, extra_mlp_layers: List[int], activation: nn.Module):
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x


class VizdoomImageProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super().__init__()
        self.common_dim = common_dim
        self.features = VizdoomConvEncoder()
        self.linear = nn.Linear(self.features.encoder_out_size, self.common_dim)
        self.finetune = finetune

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.finetune is not None and self.finetune != "image":
            with torch.no_grad():
                x = self.features(x)
                x = self.linear(x)
                return x
        x = self.features(x)
        x = self.linear(x)
        return x


class VizdoomBaseSoundEncoder(VizdoomEncoder, ABC):
    def __init__(self, sampling_rate=22050, fps=35, frame_skip=4):
        super(VizdoomBaseSoundEncoder, self).__init__()
        self.sampling_rate = sampling_rate
        self.FPS = fps
        self.frame_skip = frame_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # left side
        if isinstance(x, List):
            x = x[0]
        left = x[:, :, 0]
        left = self.encode_single_channel(left)
        # right side
        right = x[:, :, 1]
        right = self.encode_single_channel(right)
        return torch.cat((left, right), dim=1)

    @abstractmethod
    def encode_single_channel(self, data: torch.Tensor) -> torch.Tensor:
        pass


class VizdoomFFTEncoder(VizdoomBaseSoundEncoder):
    def __init__(self, obs_shape=(2520, 2), sampling_rate=22050, fps=35, frame_skip=4):
        super(VizdoomFFTEncoder, self).__init__()
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
        self.encoder_out_size = calc_num_elements(self, obs_shape)

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


def create_mlp(layer_sizes: List[int], input_size: int, activation: nn.Module) -> nn.Module:
    """Sequential fully connected layers."""
    layers = []
    for i, size in enumerate(layer_sizes):
        layers.extend([fc_layer(input_size, size), activation])
        input_size = size

    if len(layers) > 0:
        return nn.Sequential(*layers)
    else:
        return nn.Identity()


def fc_layer(in_features: int, out_features: int, bias=True, spec_norm=False) -> nn.Module:
    layer = nn.Linear(in_features, out_features, bias)
    if spec_norm:
        layer = spectral_norm(layer)

    return layer


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements


def nonlinearity(non_linear, inplace: bool = False) -> nn.Module:
    if non_linear == "elu":
        return nn.ELU(inplace=inplace)
    elif non_linear == "relu":
        return nn.ReLU(inplace=inplace)
    elif non_linear == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"Unknown {non_linear=}")
