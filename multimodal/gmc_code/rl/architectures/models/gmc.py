from typing import Optional

from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Optimizer

from gmc_code.rl.architectures.models.gmc_networks import *
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from PIL import ImageFilter
from torchvision.transforms.functional import to_pil_image
import math
import random

class GMC(LightningModule):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(GMC, self).__init__()

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None
        self.finetune = None

    def encode(self, x, sample=False):

        # If we have complete observations
        if None not in x:
            return self.encoder(self.processors[-1](x))
        else:
            latent_representations = []
            for id_mod in range(len(x)):
                if x[id_mod] is not None:
                    latent_representations.append(self.encoder(self.processors[id_mod](x[id_mod])))

            # Take the average of the latent representations
            latent = torch.stack(latent_representations, dim=0).mean(0)
            return latent

    def forward(self, x):

        # Forward pass through the modality specific encoders
        batch_representations = []
        for processor_idx in range(len(self.processors) - 1):
            mod_representations = self.encoder(
                self.processors[processor_idx](x[processor_idx])
            )
            batch_representations.append(mod_representations)

        # Forward pass through the joint encoder
        # if self.finetune is not None:
        #     with torch.no_grad():
        #         joint_representation = self.encoder(self.processors[-1](x))
        # else:
        #     joint_representation = self.encoder(self.processors[-1](x))
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations

    def infonce(self, batch_representations, temperature, batch_size):
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            # Negative pairs: everything that is not in the current joint-modality pair
            out_joint_mod = torch.cat(
                [batch_representations[-1], batch_representations[mod]], dim=0
            )
            # [2*B, 2*B]
            sim_matrix_joint_mod = torch.exp(
                torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
            )
            # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
            mask_joint_mod = (
                    torch.ones_like(sim_matrix_joint_mod)
                    - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
            ).bool()
            # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
            sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
                mask_joint_mod
            ).view(2 * batch_size, -1)

            # Positive pairs: cosine loss joint-modality
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            # [2*B]
            pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict
    

    def infonce_with_joints_as_negatives(
            self, batch_representations, temperature, batch_size
    ):
        # Similarity among joints, [B, B]
        sim_matrix_joints = torch.exp(
            torch.mm(
                batch_representations[-1], batch_representations[-1].t().contiguous()
            )
            / temperature
        )
        # Mask out the diagonals, [B, B]
        mask_joints = (
                torch.ones_like(sim_matrix_joints)
                - torch.eye(batch_size, device=sim_matrix_joints.device)
        ).bool()
        # Remove diagonals and resize, [B, B-1]
        sim_matrix_joints = sim_matrix_joints.masked_select(mask_joints).view(
            batch_size, -1
        )

        # compute loss - for each pair joint-modality
        # Cosine loss on positive pairs
        joint_mod_loss_sum = 0
        for mod in range(len(batch_representations) - 1):
            pos_sim_joint_mod = torch.exp(
                torch.sum(
                    batch_representations[-1] * batch_representations[mod], dim=-1
                )
                / temperature
            )
            loss_joint_mod = -torch.log(
                pos_sim_joint_mod / sim_matrix_joints.sum(dim=-1)
            )
            joint_mod_loss_sum += loss_joint_mod

        loss = torch.mean(joint_mod_loss_sum)
        tqdm_dict = {"loss": loss}
        return loss, tqdm_dict

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim

        tqdm_dict['weighted_loss'] = loss
        return loss, tqdm_dict

    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0].shape[0]

        # Forward pass through the encoders
        batch_representations = self.forward(data)
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
        else:
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim
        tqdm_dict['weighted_loss'] = loss
        return tqdm_dict

        # --------------- FLOP counting hooks (they fire in PL 1.5) -------

    # def on_before_backward(self, loss):
    #     # start counting *exactly* when Lightning is about to call backward()
    #     self._fc = FlopCounterMode()
    #     self._fc.__enter__()
    #
    # def on_after_backward(self):
    #     # stop counter, log, dispose
    #     flops = self._fc.get_total_flops()
    #     self._fc.__exit__(None, None, None)
    #     print("flops", flops)
    #     self.log("bwd_flops", flops, on_step=True, prog_bar=True)


class PendulumGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(PendulumGMC, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )

        self.image_processor = PendulumImageProcessor(common_dim=common_dim, finetune=finetune)
        self.sound_processor = PendulumSoundProcessor(common_dim=common_dim)
        self.joint_processor = PendulumJointProcessor(common_dim=common_dim, finetune=finetune)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type
        if finetune is not None:
            if finetune != "image":
                self.image_processor.eval()
                freeze_module(self.image_processor)
            elif finetune != "sound":
                self.sound_processor.eval()
                freeze_module(self.sound_processor)

        self.encoder = PendulumCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )
        if finetune is not None:
            self.encoder.eval()
            freeze_module(self.encoder)
        self.finetune = finetune


class HyperhotGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(HyperhotGMC, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )

        self.image_processor = HyperhotImageProcessor(common_dim=common_dim, finetune=finetune)
        self.sound_processor = HyperhotSoundProcessor(common_dim=common_dim)
        self.joint_processor = HyperhotJointProcessor(common_dim=common_dim, finetune=finetune)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type
        if finetune is not None:
            if finetune != "image":
                self.image_processor.eval()
                freeze_module(self.image_processor)
            elif finetune != "sound":
                self.sound_processor.eval()
                freeze_module(self.sound_processor)

        self.encoder = HyperhotCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )
        if finetune is not None:
            self.encoder.eval()
            freeze_module(self.encoder)
        self.finetune = finetune


class PendulumSoundOnly(PendulumGMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(PendulumGMC, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )
        self.sound_processor = PendulumSoundProcessor(common_dim=common_dim)
        self.image_processor = None
        self.joint_processor = None
        self.processors = [
            self.sound_processor,
        ]
        self.loss_type = loss_type
        self.encoder = PendulumCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )

    # def encode(self, x, sample=False):
    #     return self.encoder(self.processors[-1](x[1]))


class PendulumNullModel(nn.Identity, nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input[1]

    def encode(self, input: Tensor) -> Tensor:
        return input[1]

    def freeze(self):
        pass


class VizdoomGMC(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(VizdoomGMC, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )

        self.image_processor = VizdoomImageProcessor(common_dim=common_dim, finetune=finetune)
        self.sound_processor = VizdoomSoundProcessor(common_dim=common_dim)
        self.joint_processor = VizdoomJointProcessor(common_dim=common_dim, finetune=finetune)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type
        if finetune is not None:
            if finetune != "image":
                self.image_processor.eval()
                freeze_module(self.image_processor)
            elif finetune != "sound":
                self.sound_processor.eval()
                freeze_module(self.sound_processor)

        self.encoder = HyperhotCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )
        if finetune is not None:
            self.encoder.eval()
            freeze_module(self.encoder)


class GMCNew(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(GMCNew, self).__init__(name, common_dim, latent_dim, loss_type="infonce")

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None
        self.finetune = None
        # self.loss_weights = None
        self.augmentations = None
        # self.sound_reconstruct = None

    def forward(self, x):

        # Forward pass through the modality specific encoders
        batch_representations = []
        mod_presentations = []
        for processor_idx in range(len(self.processors) - 1):
            # print(x[processor_idx].shape)
            mod_presentations.append(
                self.processors[processor_idx](x[processor_idx]))
            mod_fused_representations = self.encoder(
                mod_presentations[-1]
            )
            batch_representations.append(mod_fused_representations)

        # Forward pass through the joint encoder
        # if self.finetune is not None:
        #     with torch.no_grad():
        #         joint_representation = self.encoder(self.processors[-1](x))
        # else:
        #     joint_representation = self.encoder(self.processors[-1](x))
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations, mod_presentations

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0][0].shape[0]
        data_augmented = data[0]
        data_original = data[1]
        # Forward pass through the encoders
        batch_representations, mod_representations = self.forward(data_augmented)

        batch_representations_original, mod_representations_original = self.forward(data_original)

        # mixed_representations = batch_representations_original[:-1] + [batch_representations[-1]]
        mixed_representations = batch_representations[:-1] + [batch_representations_original[-1]]

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss_original, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce_with_joints_as_negatives(mixed_representations, temperature, batch_size)
        else:
            loss_original, tqdm_dict = self.infonce(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce(mixed_representations, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        # reconstructed_sound_latent = self.sound_latent_reconstruct(batch_representations[1])
        # reconstructed_sound = self.sound_reconstruct(batch_representations[1])
        # latent_reconstruction_loss = F.mse_loss(data[0][1], reconstructed_sound)
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim
        # tqdm_dict["reconstruction_loss_sound"] = latent_reconstruction_loss
        tqdm_dict["gmc_loss"] = loss
        # loss = self.loss_weights(loss, loss2)
        loss = loss_original + loss + loss2
        tqdm_dict['weighted_loss'] = loss
        return loss, tqdm_dict
    
    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0][0].shape[0]
        data_augmented = data[0]
        data_original = data[1]
        # Forward pass through the encoders
        batch_representations, mod_representations = self.forward(data_augmented)

        batch_representations_original, mod_representations_original = self.forward(data_original)

        mixed_representations = batch_representations_original[:-1] + [batch_representations[-1]]
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss_original, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce_with_joints_as_negatives(mixed_representations, temperature, batch_size)
        else:
            loss_original, tqdm_dict = self.infonce(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce(mixed_representations, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        # reconstructed_sound = self.sound_reconstruct(batch_representations[1])
        # latent_reconstruction_loss = F.mse_loss(data[0][1], reconstructed_sound)
        # loss2, loss3 = loss2
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim
        # tqdm_dict["reconstruction_loss_sound"] = latent_reconstruction_loss
        tqdm_dict["gmc_loss"] = loss
        # loss = self.loss_weights(loss, loss2)
        loss = loss_original + loss + loss2
        tqdm_dict['weighted_loss'] = loss
        return tqdm_dict
    
class VizdoomGMCNew(GMCNew):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(VizdoomGMCNew, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )

        self.image_processor = VizdoomImageProcessor(common_dim=common_dim, finetune=finetune)
        self.sound_processor = VizdoomSoundProcessor(common_dim=common_dim)
        self.joint_processor = VizdoomJointProcessor(common_dim=common_dim, finetune=finetune)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type
        if finetune is not None:
            if finetune != "image":
                self.image_processor.eval()
                freeze_module(self.image_processor)
            elif finetune != "sound":
                self.sound_processor.eval()
                freeze_module(self.sound_processor)

        self.encoder = HyperhotCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )
        if finetune is not None:
            self.encoder.eval()
            freeze_module(self.encoder)
        # self.loss_weights = AutomaticWeightedLoss(3)

class GMCNew2(GMC):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(GMCNew2, self).__init__(name, common_dim, latent_dim, loss_type="infonce")

        self.name = name
        self.common_dim = common_dim
        self.latent_dim = latent_dim
        self.loss_type = loss_type

        self.image_processor = None
        self.label_processor = None
        self.joint_processor = None
        self.processors = [
            self.image_processor,
            self.label_processor,
            self.joint_processor,
        ]

        self.encoder = None
        self.finetune = None
        # self.loss_weights = None
        self.augmentations = None
        # self.sound_reconstruct = None

    def forward(self, x):

        # Forward pass through the modality specific encoders
        batch_representations = []
        mod_presentations = []
        for processor_idx in range(len(self.processors) - 1):
            # print(x[processor_idx].shape)
            mod_presentations.append(
                self.processors[processor_idx](x[processor_idx]))
            mod_fused_representations = self.encoder(
                mod_presentations[-1]
            )
            batch_representations.append(mod_fused_representations)

        # Forward pass through the joint encoder
        # if self.finetune is not None:
        #     with torch.no_grad():
        #         joint_representation = self.encoder(self.processors[-1](x))
        # else:
        #     joint_representation = self.encoder(self.processors[-1](x))
        joint_representation = self.encoder(self.processors[-1](x))
        batch_representations.append(joint_representation)
        return batch_representations, mod_presentations

    def training_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0][0].shape[0]
        data_augmented = data[0]
        data_original = data[1]
        # Forward pass through the encoders
        batch_representations, mod_representations = self.forward(data_augmented)

        batch_representations_original, mod_representations_original = self.forward(data_original)

        # mixed_representations = batch_representations_original[:-1] + [batch_representations[-1]]
        mixed_representations = batch_representations[:-1] + [batch_representations_original[-1]]
        mixed_representations_2 = batch_representations_original[:-1] + [batch_representations[-1]]

        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss_original, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce_with_joints_as_negatives(mixed_representations, temperature, batch_size)
            loss3, tqdm_dict3 = self.infonce_with_joints_as_negatives(mixed_representations_2, temperature, batch_size)
        else:
            loss_original, tqdm_dict = self.infonce(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce(mixed_representations, temperature, batch_size)
            loss3, tqdm_dict3 = self.infonce(mixed_representations_2, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        # reconstructed_sound_latent = self.sound_latent_reconstruct(batch_representations[1])
        # reconstructed_sound = self.sound_reconstruct(batch_representations[1])
        # latent_reconstruction_loss = F.mse_loss(data[0][1], reconstructed_sound)
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim
        # tqdm_dict["reconstruction_loss_sound"] = latent_reconstruction_loss
        tqdm_dict["gmc_loss"] = loss
        # loss = self.loss_weights(loss, loss2)
        loss = loss_original + loss + loss2 + loss3
        tqdm_dict['weighted_loss'] = loss
        return loss, tqdm_dict
    
    def validation_step(self, data, train_params):

        temperature = train_params["temperature"]
        batch_size = data[0][0].shape[0]
        data_augmented = data[0]
        data_original = data[1]
        # Forward pass through the encoders
        batch_representations, mod_representations = self.forward(data_augmented)

        batch_representations_original, mod_representations_original = self.forward(data_original)

        mixed_representations = batch_representations_original[:-1] + [batch_representations[-1]]
        mixed_representations_2 = batch_representations_original[:-1] + [batch_representations[-1]]
        # Compute contrastive loss
        if self.loss_type == "infonce_with_joints_as_negatives":
            loss_original, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce_with_joints_as_negatives(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce_with_joints_as_negatives(mixed_representations, temperature, batch_size)
            loss3, tqdm_dict3 = self.infonce_with_joints_as_negatives(mixed_representations_2, temperature, batch_size)
        else:
            loss_original, tqdm_dict = self.infonce(batch_representations_original, temperature, batch_size)
            loss, tqdm_dict = self.infonce(batch_representations, temperature, batch_size)
            loss2, tqdm_dict2 = self.infonce(mixed_representations, temperature, batch_size)
            loss3, tqdm_dict3 = self.infonce(mixed_representations_2, temperature, batch_size)
        image_sound_sim = F.cosine_similarity(batch_representations[0], batch_representations[1], dim=1).mean()
        image_joint_sim = F.cosine_similarity(batch_representations[0], batch_representations[2], dim=1).mean()
        sound_joint_sim = F.cosine_similarity(batch_representations[1], batch_representations[2], dim=1).mean()
        # reconstructed_sound = self.sound_reconstruct(batch_representations[1])
        # latent_reconstruction_loss = F.mse_loss(data[0][1], reconstructed_sound)
        # loss2, loss3 = loss2
        tqdm_dict["image_sound_sim"] = image_sound_sim
        tqdm_dict["image_joint_sim"] = image_joint_sim
        tqdm_dict["sound_joint_sim"] = sound_joint_sim
        # tqdm_dict["reconstruction_loss_sound"] = latent_reconstruction_loss
        tqdm_dict["gmc_loss"] = loss
        # loss = self.loss_weights(loss, loss2)
        loss = loss_original + loss + loss2 + loss3
        tqdm_dict['weighted_loss'] = loss
        return tqdm_dict
    
class VizdoomGMCNew2(GMCNew2):
    def __init__(self, name, common_dim, latent_dim, loss_type="infonce", finetune=None):
        super(VizdoomGMCNew2, self).__init__(
            name, common_dim, latent_dim, loss_type="infonce"
        )

        self.image_processor = VizdoomImageProcessor(common_dim=common_dim, finetune=finetune)
        self.sound_processor = VizdoomSoundProcessor(common_dim=common_dim)
        self.joint_processor = VizdoomJointProcessor(common_dim=common_dim, finetune=finetune)
        self.processors = [
            self.image_processor,
            self.sound_processor,
            self.joint_processor,
        ]
        self.loss_type = loss_type
        if finetune is not None:
            if finetune != "image":
                self.image_processor.eval()
                freeze_module(self.image_processor)
            elif finetune != "sound":
                self.sound_processor.eval()
                freeze_module(self.sound_processor)

        self.encoder = HyperhotCommonEncoder(
            common_dim=common_dim, latent_dim=latent_dim
        )
        if finetune is not None:
            self.encoder.eval()
            freeze_module(self.encoder)
        # self.loss_weights = AutomaticWeightedLoss(3)
