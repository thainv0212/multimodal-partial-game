import sys
import torch
from torchvision.transforms.functional import to_pil_image
import dill

path = sys.argv[1]
path_new = sys.argv[2]

image_states, sound_states, actions, rewards, \
            next_image_states, next_sound_states, done, sound_normalization = torch.load(path)

image_states = [to_pil_image(i[0].byte()) for i in image_states]
next_image_states = [to_pil_image(i[0].byte()) for i in next_image_states]

dill.dump([image_states, sound_states, actions, rewards, \
            next_image_states, next_sound_states, done, sound_normalization], path_new)