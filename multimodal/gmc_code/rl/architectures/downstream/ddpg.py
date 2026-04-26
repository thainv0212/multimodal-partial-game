import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.lightning import LightningModule
from torch.nn.modules.module import T

from gmc_code.rl.architectures.models.gmc import PendulumSoundOnly


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=256, hidden2=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=256, hidden2=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DDPG(LightningModule):
    def __init__(self, n_states, n_actions, layer_sizes):
        super(DDPG, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_layers = layer_sizes[0]
        self.critic_layers = layer_sizes[1]

        self.actor = Actor(n_states, n_actions)
        self.actor_target = Actor(n_states, n_actions)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic(self.n_states, self.n_actions)
        self.critic_target = Critic(self.n_states, self.n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, latent):
        return self.actor(latent).squeeze(0).detach().cpu().numpy()


class DDPG2(LightningModule):
    def __init__(self, n_states, n_actions, layer_sizes, finetune_encoder=False):
        super(DDPG2, self).__init__()
        # self.encoder = PendulumSoundOnly(None, 64, n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_layers = layer_sizes[0]
        self.critic_layers = layer_sizes[1]

        self.actor = Actor2(n_states, n_actions)
        self.actor_target = Actor2(n_states, n_actions)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic2(self.n_states, self.n_actions)
        self.critic_target = Critic2(self.n_states, self.n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        self.finetune_encoder = finetune_encoder
        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_target.load_state_dict(self.critic.state_dict())

    def select_action(self, latent):
        return self.actor(self.encoder(latent)).squeeze(0).detach().cpu().numpy()

    # def train(self, mode: bool = True):
    #     super(DDPG2, self).eval()
    #     self.critic.encoder.train(mode)
    #     self.actor.encoder.train(mode)
    #     return self


class Actor2(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=256, hidden2=256):
        super(Actor2, self).__init__()
        self.encoder = PendulumSoundOnly(None, 64, n_states)
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)
        self.init_weights()

    def init_weights(self):
        # self.encoder.init_weights()
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        if x.device != self.encoder.device:
            x = x.to(self.encoder.device)
        out = self.fc1(self.encoder(x)[0])
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        out = F.tanh(out)
        return out


class Critic2(nn.Module):
    def __init__(self, n_states, n_actions, hidden1=256, hidden2=256):
        super(Critic2, self).__init__()
        self.encoder = PendulumSoundOnly(None, 64, n_states)
        self.fc1 = nn.Linear(n_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + n_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, xs):
        x, a = xs
        out = self.fc1(self.encoder(x)[0])
        out = F.relu(out)
        out = self.fc2(torch.cat([out, a], 1))
        out = F.relu(out)
        out = self.fc3(out)
        return out


class DDPG3(DDPG2):
    def  __init__(self, n_states, n_actions, layer_sizes, model_file=None, finetune_encoder=False):
        super(DDPG3, self).__init__(n_states, n_actions, layer_sizes, finetune_encoder)
        # self.encoder = PendulumSoundOnly(None, 64, n_states)
        self.n_states = n_states
        self.n_actions = n_actions
        self.actor_layers = layer_sizes[0]
        self.critic_layers = layer_sizes[1]

        self.actor = Actor2(n_states, n_actions)
        self.actor_target = Actor2(n_states, n_actions)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        self.critic = Critic2(self.n_states, self.n_actions)
        self.critic_target = Critic2(self.n_states, self.n_actions)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()
        if model_file is not None:
            model_data = torch.load(model_file)
            incompatible_params = self.load_state_dict(model_data['state_dict'], strict=False)
            print(incompatible_params)
        self.finetune_encoder = finetune_encoder

    def train_encoder(self):
        self.eval()
        self.critic.encoder.train()
        self.actor.encoder.train()

    def train(self, mode: bool = True):
        # self.eval()
        # for p in self.children():
        #     p.eval()
        super().train(False)
        freeze_module(self)
        unfreeze_module(self.critic.encoder)
        unfreeze_module(self.actor.encoder)
        self.critic.encoder.train(mode)
        self.actor.encoder.train(mode)
        return self

    def load_file(self, filename):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict['state_dict'], strict=False)


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module):
    for param in module.parameters():
        param.requires_grad = True
