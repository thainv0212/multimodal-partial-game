import torch
from torch import nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder, make_img_encoder, make_sound_encoder
from sample_factory.model.model_utils import nonlinearity
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from sf_examples.vizdoom.doom.gmc.gmc import VizdoomGMC, VizdoomGMC2Wrapper, VizdoomGMCWrapper, VizdoomGMC3Wrapper, VizdoomGMC4Wrapper, VizdoomGMC5Wrapper, VizdoomGMC6Wrapper, VizdoomGMC7Wrapper, \
VizdoomGMC8Wrapper, VizdoomGMC9Wrapper, VizdoomGMC11Wrapper, VizdoomGMC12Wrapper, VizdoomGMC13Wrapper, VizdoomGMC14Wrapper, VizdoomGMC15Wrapper, VizdoomGMC16Wrapper
from sf_examples.vizdoom.doom.comm.comm import VizdoomCoMMWrapper


class VizdoomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)

        # reuse the default image encoder
        self.basic_encoder = make_img_encoder(cfg, obs_space["img"] if "measurements" not in obs_space.keys() else
        obs_space["obs"]["img"])
        if self.basic_encoder is not None:
            self.encoder_out_size = self.basic_encoder.get_out_size()
        else:
            self.encoder_out_size = 0
        self.sound_encoder = None
        if cfg.use_sound:
            self.sound_encoder = make_sound_encoder(cfg,
                                                    obs_space["sound"] if "measurements" not in obs_space.keys() else
                                                    obs_space["obs"]["sound"])
            self.encoder_out_size += self.sound_encoder.get_out_size()

        self.measurements_head = None
        if "measurements" in list(obs_space.keys()):
            self.measurements_head = nn.Sequential(
                nn.Linear(obs_space["measurements"].shape[0], 128),
                nonlinearity(cfg),
                nn.Linear(128, 128),
                nonlinearity(cfg),
            )
            measurements_out_size = calc_num_elements(self.measurements_head, obs_space["measurements"].shape)
            self.encoder_out_size += measurements_out_size

        log.debug("Policy head output size: %r", self.get_out_size())

    def forward(self, obs_dict):
        if self.basic_encoder is not None:
            x = self.basic_encoder(obs_dict["img"])
        else:
            x = None

        if self.measurements_head is not None:
            measurements = self.measurements_head(obs_dict["measurements"].float())
            if x is not None:
                x = measurements
            else:
                x = torch.cat((x, measurements), dim=1)
        if self.sound_encoder is not None:
            sound = self.sound_encoder(obs_dict["sound"].float())
            if x is not None:
                x = torch.cat((x, sound.flatten(start_dim=1)), dim=1)
            else:
                x = sound.flatten(start_dim=1)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size


def make_vizdoom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    if cfg.encoder_type == "normal":
        return VizdoomEncoder(cfg, obs_space)
    elif cfg.encoder_type == "gmc":
        return VizdoomGMCWrapper(cfg, "gmc")
    elif cfg.encoder_type == "gmcnew":
        return VizdoomGMC11Wrapper(cfg, "gmc11")
    elif cfg.encoder_type == "gmcnew2":
        return VizdoomGMC12Wrapper(cfg, "gmc12")
    elif cfg.encoder_type == "comm":
        return VizdoomCoMMWrapper(cfg, adapter=2)

