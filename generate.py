import argparse
from pathlib import Path

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import numpy as np

def generate(args, g_ema, device, mean_latent):

    args.out_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, latents = g_ema(
                [sample_z],  return_latents=args.save_latents, truncation=args.truncation, truncation_latent=mean_latent
            )

            if latents is not None:
                for ind, latent in enumerate(latents):
                    global_ind = i * args.sample + ind
                    np.save(
                        str(args.out_dir.joinpath(f"{str(global_ind).zfill(6)}.npy")),
                        latent.cpu().numpy()
                        )


            utils.save_image(
                sample,
                str(args.out_dir.joinpath(f"{str(i).zfill(6)}.png")),
                nrow=int(np.sqrt(args.sample)),
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
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

    parser.add_argument(
        "--out_dir",
        type=Path,
        default="outputs",
        help="path to generated outputs",
    )

    parser.add_argument(
        "--save_latents",
        action='store_true',
        help='Whether to save W vectors'
    )

    parser.add_argument("--map_layers", type=int, help="num of mapping layers", default=8)

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = args.map_layers

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)
