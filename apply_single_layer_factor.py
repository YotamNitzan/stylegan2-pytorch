import argparse

import torch
from torchvision import utils

from model import Generator

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.5, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--layer",
        type=int,
    )

    args = parser.parse_args()
    from pathlib import Path

    Path(args.out_dir).mkdir(exist_ok=True)

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)
    import numpy as np

    alphas = np.linspace(-3, 3, 10)
    imgs = []
    for alpha in alphas:
        if args.layer:
            wp_latent = g.get_wp([latent],
                                 truncation=args.truncation,
                                 truncation_latent=trunc,
                                 input_is_latent=True)
            wp_latent[:, args.layer] += alpha * direction

            img, _ = g.forward_from_wp(wp_latent, truncation_latent=trunc, truncation=args.truncation)
        else:
            img, _ = g(
                [latent + alpha * direction],
                truncation=args.truncation,
                truncation_latent=trunc,
                input_is_latent=True,
            )
        imgs.append(img)

    if args.layer:
        dst_path = f"{args.out_dir}/index-{args.index}_degree-{args.degree}_layer_{args.layer}.jpg"
    else:
        dst_path = f"{args.out_dir}/index-{args.index}_degree-{args.degree}.jpg"

    for i, batch in enumerate(imgs):
        for sample_idx, img in enumerate(batch):
            dst_path = f"{args.out_dir}/index-{args.index}_degree-{args.degree}_sample-{sample_idx:03}_index-{i}.jpg"
            utils.save_image(img, dst_path, normalize=True, range=(-1,1), nrow=1)

    grid = utils.save_image(
        torch.cat(imgs, 0),
        dst_path,
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )
