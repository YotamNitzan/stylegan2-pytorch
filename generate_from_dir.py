# Taken from LARGE, property of Yotam Nitzan.

import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader


def get_latents_from_dir(latents_dir, latents_suffix, is_wp=True):
    if latents_suffix == '.npy':
        latents = [(x.name, np.float32(np.load(x))) for x in latents_dir.iterdir() if x.suffix == latents_suffix]
    else:
        latents = [(x.name, torch.load(x).float()) for x in latents_dir.iterdir() if x.suffix == latents_suffix]

    if is_wp:
        latents = [(lat_name, lat_val.squeeze()) for (lat_name, lat_val) in latents]

    return latents


def get_image_path_from_latent_path(out_dir, latent_path):
    return out_dir.joinpath(Path(latent_path).with_suffix('.jpg'))


def generate_from_dir(args, g_ema, device, mean_latent):
    w_samples = get_latents_from_dir(args.latents_input_path, args.latents_suffix)
    w_samples = DataLoader(w_samples, batch_size=args.batch_size)

    for latent_filenames, latents in tqdm(w_samples):
        latents = latents.to(device)
        images, _ = g_ema.forward_from_wp(
            latents, truncation=args.truncation, randomize_noise=False,
            truncation_latent=mean_latent)

        # images = F.interpolate(images, size=256)  # TODO: remove?
        for latent_fn, image in zip(latent_filenames, images):
            utils.save_image(
                image,
                get_image_path_from_latent_path(args.out_dir, latent_fn),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )

    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument("--map_layers", type=int, help="num of mapping layers", default=8)

    parser.add_argument(
        "--latents_input_path",
        type=Path,
        help="path to latent codes"
    )

    parser.add_argument(
        "--out_dir",
        type=Path,
        help="path to output images"
    )
    parser.add_argument(
        "--latents_suffix",
        type=str,
        choices=['.npy', '.pt'],
        default='.npy'
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Batch size for generation",
    )

    parser.add_argument(
        "--truncation_layer",
        type=int,
        default=None,
        help="Until what layer to apply truncation. If not specified, applies on all layers.",
    )
    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = args.map_layers

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate_from_dir(args, g_ema, device, mean_latent)
