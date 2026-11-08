"""
Build a small external test set from Tiny-GenImage (Hugging Face) for
cross_dataset_eval.py.   Run this on YOUR computer / Kaggle notebook (needs internet).

    pip install huggingface_hub pyarrow pillow
    python prepare_genimage_subset.py --out_dir genimage_subset --per_generator 500

Output layout (what cross_dataset_eval.py expects):
    genimage_subset/real/*.<ext>
    genimage_subset/fake/<generator>/*.<ext>

Notes
- Only the VALIDATION split (about 1.75 GB) is downloaded: it was never part of your training.
- Original image bytes are saved untouched (no re-encoding), so file-format
  statistics stay honest. The script prints the format mix per class: if real
  images are mostly JPEG and fakes mostly PNG (or the reverse), a detector can
  score well by reading the format. Report that in the paper if it happens.
- Real images are sampled to match the total number of fake images.
- Dataset: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage (CC-BY-NC-SA-4.0,
  community upload of the GenImage benchmark, Zhu et al. 2023). Cite GenImage [27].
"""
import argparse
import collections
import io
import os
import random

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "TheKernel01/Tiny-GenImage"
GEN_NAMES = ["Real", "ADM", "BigGAN", "GLIDE", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="genimage_subset")
    ap.add_argument("--per_generator", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="validation")
    a = ap.parse_args()
    random.seed(a.seed)

    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
                   if f.startswith(f"data/{a.split}") and f.endswith(".parquet"))
    print("parquet files:", files)

    # pass 1: collect (file, row_group, row) indices per generator, no images loaded
    index = collections.defaultdict(list)
    paths = {}
    for f in files:
        paths[f] = hf_hub_download(REPO, f, repo_type="dataset")
        pf = pq.ParquetFile(paths[f])
        for rg in range(pf.num_row_groups):
            gens = pf.read_row_group(rg, columns=["generator"]).column("generator").to_pylist()
            for i, g in enumerate(gens):
                name = GEN_NAMES[g] if isinstance(g, int) else str(g)
                index[name].append((f, rg, i))
    print({k: len(v) for k, v in index.items()})

    n_fake_total = 0
    chosen = {}
    for name, items in index.items():
        if name == "Real":
            continue
        random.shuffle(items)
        chosen[name] = items[:a.per_generator]
        n_fake_total += len(chosen[name])
    real_items = index["Real"][:]
    random.shuffle(real_items)
    chosen["Real"] = real_items[:n_fake_total]       # balance real vs fake overall

    # pass 2: write images (original bytes)
    fmt_count = collections.defaultdict(collections.Counter)
    for name, items in chosen.items():
        folder = os.path.join(a.out_dir, "real" if name == "Real" else os.path.join("fake", name))
        os.makedirs(folder, exist_ok=True)
        by_rg = collections.defaultdict(list)
        for f, rg, i in items:
            by_rg[(f, rg)].append(i)
        k = 0
        for (f, rg), rows in by_rg.items():
            tbl = pq.ParquetFile(paths[f]).read_row_group(rg, columns=["image"])
            imgs = tbl.column("image").to_pylist()
            for i in rows:
                data = imgs[i]["bytes"]
                fmt = (Image.open(io.BytesIO(data)).format or "UNK").lower()
                ext = "jpg" if fmt == "jpeg" else fmt
                with open(os.path.join(folder, f"{name}_{k:05d}.{ext}"), "wb") as fh:
                    fh.write(data)
                fmt_count["real" if name == "Real" else "fake"][fmt] += 1
                k += 1
        print(f"{name:12} wrote {k}")

    print("\nFILE-FORMAT MIX (check for a real/fake format shortcut):")
    for cls in ("real", "fake"):
        print(" ", cls, dict(fmt_count[cls]))


if __name__ == "__main__":
    main()