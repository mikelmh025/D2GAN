# usage: python gen_imgs.py runs/peergan-02Oct19-00.13.56/checkpoints/187200/g.pt 60000
#                           <generator model>                                     <num samples>
import os
from argparse import ArgumentParser

import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import models

parser = ArgumentParser()
parser.add_argument('--path', type=str, help='Path to the generator')
parser.add_argument('--num_samples', type=int, help='number of samples to generate')

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


def gen(g, num_samples=60000, latent_size=100, path="images"):
    os.makedirs(path, exist_ok=True)

    for i in range(num_samples):
        # Sample noise as generator input
        if CUDA:
            fixed_noise = torch.cuda.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        else:
            fixed_noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        fixed_noise = Variable(fixed_noise)
        gen_imgs = g(fixed_noise)

        save_image(gen_imgs.data[0], os.path.join(path, f"{i}.png"), normalize=True)


if __name__ == '__main__':
    args = parser.parse_args()
    g = models._netG()
    g.load_state_dict(torch.load(args.path))

    # g = torch.load(args.path)
    print('g',g)
    gen(g, args.num_samples)
