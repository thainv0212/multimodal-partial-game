from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from abc import ABC, abstractmethod


# Pendulum
class PendulumCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(PendulumCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class PendulumImageProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super(PendulumImageProcessor, self).__init__()
        self.common_dim = common_dim

        self.image_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.projector = nn.Linear(14400, common_dim)
        self.finetune = finetune

    def forward(self, x):
        if self.finetune is not None and self.finetune != 'image':
            with torch.no_grad():
                x = self.image_features(x)
                x = x.view(x.size(0), -1)
                return self.projector(x)
        x = self.image_features(x)
        x = x.view(x.size(0), -1)
        return self.projector(x)


class PendulumSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(PendulumSoundProcessor, self).__init__()

        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(50, common_dim)

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.snd_features(x)
        return self.projector(h)


# Pendulum
class PendulumJointProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super(PendulumJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.n_stack = 2
        self.sound_channels = 3
        self.sound_length = 2
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            Swish(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 50),
            Swish(),
            nn.Linear(50, 50),
            Swish(),
        )

        self.projector = nn.Linear(14400 + 50, common_dim)
        if finetune is not None:
            if finetune != "image":
                self.img_features.eval()
                freeze_module(self.img_features)
            if finetune != "sound":
                self.snd_features.eval()
                freeze_module(self.snd_features)
            self.projector.eval()
            freeze_module(self.projector)
        self.finetune = finetune

    def forward(self, x):

        x_img, x_snd = x[0], x[1]
        if self.finetune is not None and self.finetune != 'image':
            with torch.no_grad():
                x_img = self.img_features(x_img)
        else:
            x_img = self.img_features(x_img)
        x_img = x_img.view(x_img.size(0), -1)

        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


class HyperhotImageProcessor(nn.Module):
    """Parametrizes q(z|x).
    @param n_latents: integer
                      number of latent dimensions
    """

    def __init__(self, common_dim, finetune):
        super(HyperhotImageProcessor, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 8, 4, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0, bias=False),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU())

        self.latent_dim = common_dim

        self.fc_mu = nn.Linear(512, common_dim)
        # self.fc_logvar = nn.Linear(512, latent_dim)
        self.finetune = finetune

    def forward(self, x):
        if self.finetune is not None and self.finetune != 'image':
            with torch.no_grad():
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return self.fc_mu(x)  # , self.fc_logvar(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.fc_mu(x)  # , self.fc_logvar(x)


class HyperhotSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super(HyperhotSoundProcessor, self).__init__()

        self.n_stack = 2
        self.sound_channels = 4
        self.sound_length = 1047

        self.unrolled_sound_input = self.n_stack * self.sound_channels * self.sound_length

        self.fc1 = nn.Linear(self.unrolled_sound_input, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc31 = nn.Linear(512, common_dim)
        # self.fc32 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.unrolled_sound_input)
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc31(h)  # , self.fc32(h)


class HyperhotCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(HyperhotCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )
        # self.finetune = finetune

    def forward(self, x):
        # if self.finetune is not None:
        #     with torch.no_grad():
        #         return F.normalize(self.feature_extractor(x), dim=-1)
        return F.normalize(self.feature_extractor(x), dim=-1)


class HyperhotJointProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super(HyperhotJointProcessor, self).__init__()
        # Variables
        self.common_dim = common_dim
        # self.n_stack = 2
        # self.sound_channels = 3
        # self.sound_length = 2
        self.n_stack = 2
        self.sound_channels = 4
        self.sound_length = 1047
        self.unrolled_sound_input = (
                self.n_stack * self.sound_channels * self.sound_length
        )

        self.img_features = nn.Sequential(
            nn.Conv2d(2, 32, 8, 4, 2, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 0, bias=False),
            nn.ReLU(),
        )

        self.snd_features = nn.Sequential(
            nn.Linear(self.unrolled_sound_input, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )

        self.projector = nn.Linear(4096 + 512, common_dim)
        if finetune is not None:
            if finetune != "image":
                self.img_features.eval()
                freeze_module(self.img_features)
            if finetune != "sound":
                self.snd_features.eval()
                freeze_module(self.snd_features)
            self.projector.eval()
            freeze_module(self.projector)
        self.finetune = finetune

    def forward(self, x):

        x_img, x_snd = x[0], x[1]
        if self.finetune is not None and self.finetune != 'image':
            with torch.no_grad():
                x_img = self.img_features(x_img)
                x_img = x_img.view(x_img.size(0), -1)
        else:
            x_img = self.img_features(x_img)
            x_img = x_img.view(x_img.size(0), -1)
        x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


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
        # print(x)
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


class VizdoomCommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(VizdoomCommonEncoder, self).__init__()
        # Variables
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 128), Swish(), nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class VizdoomJointProcessor(nn.Module):
    def __init__(self, common_dim, finetune=None):
        super().__init__()
        self.common_dim = common_dim
        self.finetune = finetune
        self.img_features = VizdoomConvEncoder()
        self.snd_features = VizdoomFFTEncoder()
        self.projector = nn.Linear(1024, common_dim)
        if finetune is not None:
            if finetune != "image":
                self.img_features.eval()
                freeze_module(self.img_features)
            if finetune != "sound":
                self.snd_features.eval()
                freeze_module(self.snd_features)
            self.projector.eval()
            freeze_module(self.projector)
        self.finetune = finetune

    def forward(self, x):

        x_img, x_snd = x[0], x[1]
        if self.finetune is not None and self.finetune == "image":
            with torch.no_grad():
                x_img = self.img_features(x_img)
                x_img = x_img.view(x_img.size(0), -1)
        else:
            x_img = self.img_features(x_img)
            x_img = x_img.view(x_img.size(0), -1)

        # x_snd = x_snd.view(-1, self.unrolled_sound_input)
        x_snd = self.snd_features(x_snd)
        if self.finetune is not None and self.finetune == "image":
            with torch.no_grad():
                return self.projector(torch.cat((x_img, x_snd), dim=-1))
        return self.projector(torch.cat((x_img, x_snd), dim=-1))


class VizdoomSoundProcessor(nn.Module):
    def __init__(self, common_dim):
        super().__init__()
        self.common_dim = common_dim
        self.features = VizdoomFFTEncoder()
        self.linear = nn.Linear(512, common_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.linear(x)


"""


Extra components


"""


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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def nonlinearity(non_linear, inplace: bool = False) -> nn.Module:
    if non_linear == "elu":
        return nn.ELU(inplace=inplace)
    elif non_linear == "relu":
        return nn.ReLU(inplace=inplace)
    elif non_linear == "tanh":
        return nn.Tanh()
    else:
        raise Exception(f"Unknown {non_linear=}")


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


class SoundLatentReconstructor(nn.Module):
    def __init__(self, latent_dim, common_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 128), Swish(), nn.Linear(128, common_dim))

    def forward(self, x):
        return self.fc(x)

class SoundReconstructor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), Swish(), nn.Linear(512, 5040))
    def forward(self, x):
        output = self.fc(x)
        return output.view(-1, 2520, 2)

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0

        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
