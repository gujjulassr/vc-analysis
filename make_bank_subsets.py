#!/usr/bin/env python
"""
make_bank_subsets.py — build subsampled FAISS indexes for the bank-size curve
("how many vectors is a voice?" — zero-training probe).

Reads an existing RVC .index, randomly subsamples its entries to several sizes,
and writes one IndexFlatL2 per size (compatible with RVC/Applio inference:
supports .search and .reconstruct_n).

Usage:
  python make_bank_subsets.py --index logs/<exp>/<file>.index --out_dir bank_subsets
  # then run inference once per subset index (same chunks, same settings,
  # index_rate=1.0), and analysis_vc.py with one --system per size.
"""

import argparse, os

import faiss
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True, help="existing .index file")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--sizes", default="10,100,1000,10000",
                help="comma-separated bank sizes")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

ix = faiss.read_index(args.index)
big = ix.reconstruct_n(0, ix.ntotal)          # [N, dim] full bank
print(f"loaded {args.index}: {big.shape[0]} entries, dim={big.shape[1]}")

for n in [int(s) for s in args.sizes.split(",")]:
    n_eff = min(n, big.shape[0])
    sub = big[rng.choice(big.shape[0], n_eff, replace=False)].astype(np.float32)
    out = faiss.IndexFlatL2(big.shape[1])
    out.add(sub)
    path = os.path.join(args.out_dir, f"bank_{n_eff}.index")
    faiss.write_index(out, path)
    print(f"wrote {path}  ({n_eff} entries)")

print("\nNext: infer the SAME eval chunks once per subset index "
      "(index_rate=1.0, all else frozen), then:\n"
      "  analysis_vc.py --system size10=... --system size100=... "
      "--system size1000=... --system size10000=...")
