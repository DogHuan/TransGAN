import torch
from models import Generator, Discriminator
from PIL import Image
import numpy as np
from torchvision import transforms as tvt
from fid_score import *


def get_generator_net():
    generator = Generator(depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                    drop_rate=0.5)  # ,device = device)

    state_dict = torch.load("checkpoint/checkpoint.pth")['generator_state_dict']
    generator.load_state_dict(state_dict, strict=True)
    return generator


def get_discriminator_net():
    discriminator = Discriminator(diff_aug="translation,cutout,color", image_size=32, patch_size=4, input_channel=3,
                        num_classes=1,
                        dim=384, depth=7, heads=4, mlp_ratio=4,
                        drop_rate=0.)
    state_dict = torch.load("checkpoint/checkpoint.pth")["discriminator_state_dict"]
    discriminator.load_state_dict(state_dict, strict=True)
    return discriminator


def main():
    device = torch.device("cuda")
    generator = get_generator_net().to(device)
    discriminator = get_discriminator_net().to(device)
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'

    img = Image.open("image_file")
    img = tvt.ToTensor()(img)
    img = img.unsqueeze(0)
    img.type(torch.FloatTensor)
    img = img.cuda()

    noise = torch.FloatTensor(np.random.normal(0, 1, (1, 1024))).cuda()

    fake_img = generator(noise)

    valid = discriminator(fake_img)

    fid_score = get_fid(fid_stat, 200 , generator, num_img=5000, val_batch_size=60 * 2, latent_dim=1024,
                        writer_dict=None, cls_idx=None)

    print(f"FID score: {fid_score}")


if __name__ == "__main__":
    main()
