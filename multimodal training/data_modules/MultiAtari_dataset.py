import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset, IterableDataset
from gmc_code.rl.data_modules.utils.atari_dataset_utils import generate_dataset_filename, generate_dataset, \
    generate_hyperhot_dataset_filename, generate_vizdoom_dataset_filename
import math
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import random
from PIL import ImageFilter

# 'state', 'action', 'next_state', 'reward', 'terminal'

class MultiAtariDataModule(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.dataset,
                                                                                  scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):
            print('[MultiAtari] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config,
                                      train=True)
            train_partition = int(len(atari_full) * 0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
        if stage == 'validate' or stage is None:
            self.val_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.val_data[0][0][0].shape)
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])


class MultiHyperhotDataModule(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_hyperhot_dataset_filename(scenario=self.dataset,
                                                                                           scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):
            print('[MultiHyperhot] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config,
                                      train=True)
            train_partition = int(len(atari_full) * 0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
        if stage == 'validate' or stage is None:
            self.val_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.val_data[0][0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])


class MultiVizdoomDataModule(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_vizdoom_dataset_filename(scenario=self.dataset,
                                                                                          scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):
            print('[Vizdoom] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config,
                                      train=True)
            train_partition = int(len(atari_full) * 0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
        if stage == 'validate' or stage is None:
            self.val_data = AtariDataset(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.val_data[0][0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

class MultiVizdoomData2Module(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_vizdoom_dataset_filename(scenario=self.dataset,
                                                                                          scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):
            print('[Vizdoom] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset2(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config,
                                      train=True)
            train_partition = int(len(atari_full) * 0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset2(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
        if stage == 'validate' or stage is None:
            self.val_data = AtariDataset2(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.val_data[0][0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])
    
class MultiVizdoomData3Module(LightningDataModule):

    def __init__(self, dataset, data_dir, scenario_config, data_config):
        super().__init__()

        # DataModule variables;
        self.dataset = str.lower(dataset)
        self.data_dir = data_dir
        self.data_config = data_config
        self.scenario_config = scenario_config

        # Data-specific variables - fill with setup function;
        self.train_data, self.val_data, self.test_data = None, None, None

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self.data_dir, filename))

    def prepare_data(self):
        # generate if it does not exist
        train_dataset_filename, test_dataset_filename = generate_vizdoom_dataset_filename(scenario=self.dataset,
                                                                                          scenario_cfg=self.scenario_config)
        if not self._check_exists(train_dataset_filename):
            print('[Vizdoom] Dataset not found. Generating dataset...')
            generate_dataset(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            atari_full = AtariDataset3(scenario=self.dataset, data_dir=self.data_dir, scenario_cfg=self.scenario_config,
                                      train=True)
            train_partition = int(len(atari_full) * 0.9)
            val_partition = len(atari_full) - train_partition
            self.train_data, self.val_data = random_split(atari_full, [train_partition, val_partition])
            self.dims = tuple(self.train_data[0][0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_data = AtariDataset3(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.test_data[0][0][0].shape)
        if stage == 'validate' or stage is None:
            self.val_data = AtariDataset3(scenario=self.dataset, data_dir=self.data_dir,
                                          scenario_cfg=self.scenario_config, train=False)
            self.dims = tuple(self.val_data[0][0][0].shape)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.data_config['batch_size'],
            shuffle=True,
            num_workers=self.data_config['num_workers'],
        )

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.data_config['batch_size'],
                          shuffle=False, num_workers=self.data_config['num_workers'])

class MultiAtariDataset(IterableDataset):
    """
    Iterable Dataset containing the ReplayBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer, sample_size: int = 200):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        states, actions, next_states, rewards, terminals = self.buffer.sample(self.sample_size)
        for i in range(len(terminals)):
            yield states[i], actions[i], next_states[i], rewards[i], terminals[i]


class AtariDataset(Dataset):
    def __init__(self,
                 scenario,
                 data_dir,
                 scenario_cfg,
                 train=True):

        # Get variables
        self.scenario = scenario
        self.data_dir = data_dir
        self.scenario_cfg = scenario_cfg
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.scenario,
                                                                                  scenario_cfg=self.scenario_cfg)

        # Loading data from dataset
        if train:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, train_dataset_filename), weights_only=False)
        else:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, test_dataset_filename), weights_only=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, sound)
        """
        img, sound, action, reward, next_img, next_snd, done = self._image_states[index], self._sound_states[index], \
            self._actions[index], self._rewards[index], \
            self._next_image_states[index], self._next_sound_states[index], self._done[index]

        return [img.squeeze(), sound.squeeze()], [next_img.squeeze(),
                                                  next_snd.squeeze()], action.squeeze(), reward.squeeze(), done.squeeze()

    def __len__(self):
        return len(self._image_states)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self._root, filename))

    def get_sound_normalization(self):
        return self._sound_normalization


class AtariDataset2(Dataset):
    def __init__(self,
                 scenario,
                 data_dir,
                 scenario_cfg,
                 train=True):

        # Get variables
        self.scenario = scenario
        self.data_dir = data_dir
        self.scenario_cfg = scenario_cfg
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.scenario,
                                                                                  scenario_cfg=self.scenario_cfg)

        # Loading data from dataset
        if train:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, train_dataset_filename), weights_only=False)
        else:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, test_dataset_filename), weights_only=False)
        self.augmentations = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.augmentations.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((72, 128), scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        self.augmentations.append(self.noise)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, sound)
        """
        img, sound, action, reward, next_img, next_snd, done = self._image_states[index], self._sound_states[index], \
            self._actions[index], self._rewards[index], \
            self._next_image_states[index], self._next_sound_states[index], self._done[index]

        return [self.augmentations[0](img.squeeze()), self.augmentations[1](sound.squeeze())], [self.augmentations[0](next_img.squeeze()),
                                                  self.augmentations[1](next_snd.squeeze())], action.squeeze(), reward.squeeze(), done.squeeze()

    def __len__(self):
        return len(self._image_states)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self._root, filename))

    def get_sound_normalization(self):
        return self._sound_normalization
    
    def noise(self, x, std=0.1):
        # return [x_ + np.random.randn(*x_.shape).astype(np.float32) * std for x_ in x]
        # return [x_ + torch.randn(*x_.shape).type(torch.float32) * std for x_ in x]
        return x + torch.randn(*x.shape).type(torch.float32) * std
    

class AtariDataset3(Dataset):
    def __init__(self,
                 scenario,
                 data_dir,
                 scenario_cfg,
                 train=True):

        # Get variables
        self.scenario = scenario
        self.data_dir = data_dir
        self.scenario_cfg = scenario_cfg
        train_dataset_filename, test_dataset_filename = generate_dataset_filename(scenario=self.scenario,
                                                                                  scenario_cfg=self.scenario_cfg)

        # Loading data from dataset
        if train:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, train_dataset_filename), weights_only=False)
        else:
            self._image_states, self._sound_states, self._actions, self._rewards, \
                self._next_image_states, self._next_sound_states, self._done, self._sound_normalization = torch.load(
                os.path.join(self.data_dir, test_dataset_filename), weights_only=False)
        self.augmentations = []
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        self.augmentations.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((72, 128), scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        self.augmentations.append(self.noise)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, sound)
        """
        img, sound, action, reward, next_img, next_snd, done = self._image_states[index], self._sound_states[index], \
            self._actions[index], self._rewards[index], \
            self._next_image_states[index], self._next_sound_states[index], self._done[index]

        return [self.augmentations[0](img.squeeze()), self.augmentations[1](sound.squeeze())],[img.squeeze(), sound.squeeze()], [self.augmentations[0](next_img.squeeze()),
                                                  self.augmentations[1](next_snd.squeeze())], action.squeeze(), reward.squeeze(), done.squeeze()

    def __len__(self):
        return len(self._image_states)

    def _check_exists(self, filename):
        return os.path.exists(os.path.join(self._root, filename))

    def get_sound_normalization(self):
        return self._sound_normalization
    
    def noise(self, x, std=0.1):
        # return [x_ + np.random.randn(*x_.shape).astype(np.float32) * std for x_ in x]
        # return [x_ + torch.randn(*x_.shape).type(torch.float32) * std for x_ in x]
        return x + torch.randn(*x.shape).type(torch.float32) * std

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def pil_like_gaussian_blur(self, sigma, x):
        kernel_size = 2 * math.ceil(3 * sigma) + 1  # same as PIL's default
        kernel_size = max(kernel_size, 3)  # minimum 3x3
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)(x)
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 4:
                x = to_pil_image(x[0].byte())
            else:
                x = to_pil_image(x.byte())
        # else:
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x