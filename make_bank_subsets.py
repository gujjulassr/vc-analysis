#!/usr/bin/env python
"""
make_bank_subsets.py — build speaker memory banks of varying size K for the
capacity curve: "how many vectors does it take to represent a voice?"

Reads an existing RVC .index (ideally a PER-SPEAKER index) and writes one
IndexFlatL2 per size K (usable directly for RVC/Applio inference). Two methods:
  --method kmeans   K centroids covering the feature space   (paper version)
  --method random   K random entries                         (quick probe)

K=1 (kmeans) is the speaker's mean content vector ~= a content-space speaker
embedding; K=full is the ordinary bank. Sweeping K connects the two extremes
(emb_g = 1 vector  <-->  full bank = ~10k vectors).

Usage:
  python make_bank_subsets.py --index logs/<spk_exp>/<file>.index \
      --out_dir bank_subsets --method kmeans \
      --sizes 1,2,4,8,16,64,256,1024,4096,10000
Then infer the SAME eval chunks once per bank_K.index (index_rate FIXED, all
else frozen) -> infer_K<K>/, and analysis_vc.py with one --system K<K> each.
"""

import argparse, os

import faiss
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--index", required=True, help="existing .index file")
ap.add_argument("--out_dir", required=True)
ap.add_argument("--method", choices=["kmeans", "random"], default="kmeans")
ap.add_argument("--sizes", default="1,2,4,8,16,64,256,1024,4096,10000",
                help="comma-separated bank sizes K")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
rng = np.random.default_rng(args.seed)

ix = faiss.read_index(args.index)
big = ix.reconstruct_n(0, ix.ntotal).astype(np.float32)   # [N, dim] full bank
dim = big.shape[1]
print(f"loaded {args.index}: {big.shape[0]} entries, dim={dim}, method={args.method}")

for n in [int(s) for s in args.sizes.split(",")]:
    n_eff = min(n, big.shape[0])
    if args.method == "kmeans" and n_eff < big.shape[0]:
        km = faiss.Kmeans(dim, n_eff, niter=20, seed=args.seed, verbose=False)
        km.train(big)
        sub = km.centroids.reshape(n_eff, dim).astype(np.float32)
    else:                                                  # random (or K == full)
        sub = big[rng.choice(big.shape[0], n_eff, replace=False)]
    out = faiss.IndexFlatL2(dim)
    out.add(sub)
    path = os.path.join(args.out_dir, f"bank_{n_eff}.index")
    faiss.write_index(out, path)
    print(f"wrote {path}  ({n_eff} vectors)")

print("\nNext: infer the SAME eval chunks once per bank_K.index (FIXED "
      "index_rate, all else frozen) -> infer_K<K>/, then:\n"
      "  analysis_vc.py --system K1=infer_K1 --system K8=infer_K8 ... "
      "--lang <code> --whisper large-v3 --out results_capacity")
