import argparse
from pathlib import Path

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

import os
import numpy as np
import random

def generate(args, g_ema, device, mean_latent):

    args.out_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        g_ema.eval()

        if args.input_is_latent:

            boundary = np.load(args.boundary_path)
            latent_space_steps = np.linspace(-args.max_latent_step, args.max_latent_step, args.latent_steps_number)\
                .round(decimals=2)
            latent_codes = random.sample([x for x in os.listdir(args.latents_input_path) if x.endswith('npy')],
                                         args.pics)

            for i in tqdm(range(len(latent_codes))):
                sample_w = np.expand_dims(np.load(Path(args.latents_input_path) / latent_codes[i])[0], 0)

                for step in latent_space_steps:
                    sample_w_edit = torch.Tensor(sample_w + step * boundary).to(device)
                    sample, latents = g_ema(
                        [sample_w_edit], return_latents=args.save_latents, truncation=args.truncation,
                        truncation_latent=mean_latent,
                        input_is_latent=True
                    )

                    if latents is not None:
                        for ind, latent in enumerate(latents):
                            global_ind = i * args.sample + ind
                            np.save(
                                str(args.out_dir.joinpath(f"{str(global_ind).zfill(6)}_{step}.npy")),
                                latent.cpu().numpy()
                            )

                    utils.save_image(
                        sample,
                        str(args.out_dir.joinpath(f"{str(i).zfill(6)}_{step}.png")),
                        nrow=int(np.sqrt(args.sample)),
                        normalize=True,
                        range=(-1, 1),
                    )


        else:
            for i in tqdm(range(args.pics)):
                sample_z = torch.randn(args.sample, args.latent, device=device)
                sample, latents = g_ema(
                    [sample_z],  return_latents=args.save_latents, truncation=args.truncation, truncation_latent=mean_latent,
                    input_is_latent=False
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

    parser.add_argument(
        "--input_is_latent",
        action='store_true',
        help='input is w vector'
    )

    parser.add_argument(
        "--latents_input_path",
        type=Path,
        help="path to latent codes"
    )

    parser.add_argument(
        "--boundary_path",
        type=Path,
        help="boundary.npy path file for latent space edit"
    )

    parser.add_argument(
        "--max_latent_step",
        type=int,
        default=5,
        help="maximum latent space step"
    )

    parser.add_argument(
        "--latent_steps_number",
        type=int,
        default=10,
        help="number of steps for each latent input"
    )

    parser.add_argument("--map_layers", type=int, help="num of mapping layers", default=8)

    args = parser.parse_args()

    if args.input_is_latent and (args.latents_input_path is None or args.boundary_path is None):
        raise("both boudary_path and latents_input_path should be inserted if input is latent")

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
