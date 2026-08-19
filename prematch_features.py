#!/usr/bin/env python
"""
prematch_features.py — build kNN-VC / FragmentVC-style "prematched" training features.

For each utterance, replace every ContentVec frame with the mean of its k nearest
neighbors taken from the SAME speaker's OTHER utterances (the query utterance is
excluded from its own pool). Training a VC model on these features teaches it to
render audio from RETRIEVED features -- matching what happens at inference (FAISS
retrieval), which closes the train/inference gap.

Prior art to cite: kNN-VC (Interspeech 2023), FragmentVC (ICASSP 2021).

Usage:
  python prematch_features.py --filelist logs/<exp>/filelist.txt --k 4
Writes:
  <exp>/feature_prematched/<sid>_<idx>.npy
  <exp>/filelist_prematched.txt
Then train a NEW model whose training filelist is filelist_prematched.txt.
"""

import argparse, os

import numpy as np
import faiss

ap = argparse.ArgumentParser()
ap.add_argument("--filelist", required=True, help="Applio filelist.txt")
ap.add_argument("--out_dir", default=None,
                help="default: <filelist dir>/feature_prematched")
ap.add_argument("--k", type=int, default=4, help="neighbors to average per frame")
args = ap.parse_args()

exp_dir = os.path.dirname(os.path.abspath(args.filelist))
out_dir = args.out_dir or os.path.join(exp_dir, "feature_prematched")
os.makedirs(out_dir, exist_ok=True)

# filelist format:  wav | feat | f0 | f0nsf | sid   (sid = last field, feat = 2nd)
rows, by_sid = [], {}
with open(args.filelist) as f:
    for line in f:
        p = line.strip().split("|")
        if len(p) < 3:
            continue
        rows.append(p)
        by_sid.setdefault(p[-1], []).append(len(rows) - 1)
print(f"{len(rows)} utterances, {len(by_sid)} speakers")

new_lines = []
for sid, idxs in by_sid.items():
    arrs = {}
    for i in idxs:
        try:
            arrs[i] = np.load(rows[i][1]).astype(np.float32)      # [T, dim]
        except Exception as e:
            print(f"  skip (load fail) {rows[i][1]}: {e}")
    for i in idxs:
        if i not in arrs:
            continue
        q = arrs[i]
        others = [arrs[j] for j in idxs if j != i and j in arrs]
        if not others:                          # speaker has a single utterance
            pm = q
        else:
            pool = np.concatenate(others, 0)
            idx = faiss.IndexFlatL2(pool.shape[1])
            idx.add(pool)
            _, I = idx.search(q, min(args.k, len(pool)))
            pm = pool[I].mean(1)                 # [T, k, dim] -> [T, dim]
        outp = os.path.join(out_dir, f"{sid}_{i}.npy")
        np.save(outp, pm.astype(np.float32))
        p = list(rows[i]); p[1] = outp
        new_lines.append("|".join(p))
    print(f"sid {sid}: {len(idxs)} utts prematched")

out_fl = os.path.join(exp_dir, "filelist_prematched.txt")
with open(out_fl, "w") as f:
    f.write("\n".join(new_lines) + "\n")
print(f"\nwrote {out_fl}\n      {out_dir}/")
print("Next: train a NEW model on filelist_prematched.txt (copy it into a fresh "
      "exp folder as filelist.txt, or point config.data.training_files at it).")
