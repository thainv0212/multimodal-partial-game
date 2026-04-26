from torch import nn
import torch
from collections import OrderedDict
from typing import Dict, List
# Local imports
from sf_examples.vizdoom.doom.comm.base import BaseModel
from sf_examples.vizdoom.doom.comm.comm_loss import CoMMLoss, CoMMWeightedLoss, CoMMWeightedLoss2
from sf_examples.vizdoom.doom.comm.mmfusion import MMFusion
from sample_factory.model.encoder import Encoder
from sf_examples.vizdoom.doom.comm.vizdoom_encoder import VizdoomConvEncoder, VizdoomFFTEncoder
from sf_examples.vizdoom.doom.comm.input_adapters import FeaturesInputAdapter, SimpleFeaturesInputAdapter


class CoMM(BaseModel):
    """ Contrastive MultiModal learning allowing the communication between modalities
    in a single multimodal space [1].

    It encodes a pair of mulitmodal data and outputs a pair of representations through
    a single multimodal encoder.

    [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(CoMM, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = CoMMLoss(**loss_kwargs)

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor]):
        # compute features for all modalities
        all_masks = self.gen_all_possible_masks(len(x1))
        z1 = self.encoder(x1, mask_modalities=all_masks)
        z2 = self.encoder(x2, mask_modalities=all_masks)
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        return {'aug1_embed': z1,
                'aug2_embed': z2,
                "prototype": -1}

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal features from the encoder.
           Args:
                loader: Dataset loader to serve `(X, y)` tuples.
                kwargs: given to `encoder.forward()`
           Returns:
                Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor):  # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

class WeightedCoMM(CoMM):
    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(WeightedCoMM, self).__init__(encoder, projection, optim_kwargs, loss_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = CoMMWeightedLoss(**loss_kwargs)

class WeightedCoMM2(CoMM):
    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the CoMM loss.
        """
        super(WeightedCoMM2, self).__init__(encoder, projection, optim_kwargs, loss_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = CoMMWeightedLoss2(**loss_kwargs)

class VizdoomCoMMWrapper(Encoder, nn.Module):
    def __init__(self, cfg, adapter=1, model=1):
        super(VizdoomCoMMWrapper, self).__init__(cfg)
        # self.comm = VizdoomGMC(name, cfg.gmc_common_dim, cfg.gmc_latent_dim, cfg.gmc_loss_type, None,
        #                       cfg.gmc_freeze,
        #                       cfg.gmc_model_file)
        if model == 1:
            m = CoMM
        elif model == 2:
            m = WeightedCoMM
        elif model == 3:
            m = WeightedCoMM2
        else:
            raise Exception("Unknown Comm model")

        if adapter ==1:
            self.comm = m(
                encoder=MMFusion(
                    encoders=[  # Symmetric visual encoders
                        VizdoomConvEncoder(),
                        VizdoomFFTEncoder()
                    ],
                    input_adapters=[  # Pach adapters for multimodal fusion
                        FeaturesInputAdapter(n_features=512, dim_tokens=512),
                        FeaturesInputAdapter(n_features=512, dim_tokens=512)
                    ],
                    embed_dim=512
                ),
                projection=CoMM._build_mlp(512, 512, 256),
                optim_kwargs=dict(lr=3e-4, weight_decay=1e-4),
                loss_kwargs=dict(temperature=0.1)
            )
        elif adapter == 2:
            self.comm = m(
                encoder=MMFusion(
                    encoders=[  # Symmetric visual encoders
                        VizdoomConvEncoder(),
                        VizdoomFFTEncoder()
                    ],
                    input_adapters=[  # Pach adapters for multimodal fusion
                        SimpleFeaturesInputAdapter(),
                        SimpleFeaturesInputAdapter()
                    ],
                    embed_dim=512
                ),
                projection=m._build_mlp(512, 512, 256),
                optim_kwargs=dict(lr=3e-4, weight_decay=1e-4),
                loss_kwargs=dict(temperature=0.1)
            )
        self.cfg = cfg
        print("Loading CoMM model file")
        state_dict = torch.load(cfg.comm_model_file, weights_only=False)
        new_state_dict = {}
        for k,v in state_dict["state_dict"].items():
            if not k.startswith("comm."):
                new_state_dict["comm." + k] = v
            else:
                new_state_dict[k] = v
        print(new_state_dict.keys())
        # self.load_state_dict(state_dict["state_dict"])
        print(self.load_state_dict(new_state_dict))

    def encode(self, x):
        masks = []
        for mod in x:
            if mod is not None:
                masks.append(True)
            else:
                masks.append(False)
        return self.comm.encoder(x)

    def get_out_size(self) -> int:
        return 512

    def freeze_weights(self):
        self.comm.eval()
        freeze_module(self.comm)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False

