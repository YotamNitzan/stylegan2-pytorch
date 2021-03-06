import argparse
import numpy as np
import torch
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    eigvec = torch.svd(W).V.to("cpu")

    for i in range(eigvec.shape[1]):
        vec = eigvec[:, i].cpu().numpy()
        np.save(Path(args.out).joinpath(f'factor_idx_{i}.npy'), vec)
    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, Path(args.out).joinpath(f'factors.pt'))

