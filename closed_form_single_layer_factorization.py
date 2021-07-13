import argparse
from pathlib import Path
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )

    parser.add_argument(
        "--out_dir", type=str, default="layer_factors/")

    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    args = parser.parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)

    ckpt = torch.load(args.ckpt)
    modulate = {
        k: v
        for k, v in ckpt["g_ema"].items()
        if "modulation" in k and "to_rgbs" not in k and "weight" in k
    }

    for k, v in modulate.items():
        eigvecs = torch.svd(v).V.to("cpu")
        layer_name=k.replace('.', '-')
        out_path = Path(args.out_dir).joinpath(f'factors_{layer_name}.pt')
        torch.save({"ckpt": args.ckpt, "eigvec": eigvecs}, out_path)

    # weight_mat = []
    # for k, v in modulate.items():
    #     weight_mat.append(v)
    #
    # W = torch.cat(weight_mat, 0)
    # eigvec = torch.svd(W).V.to("cpu")
    #
    # torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)
