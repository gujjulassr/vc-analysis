#!/usr/bin/env python
"""
verify_prematch.py — INDEPENDENT correctness check for prematch_features.py.

This does NOT use faiss. It recomputes what every prematched frame *should* be
using plain-numpy brute-force k-NN, then compares to what prematch_features.py
actually wrote. If the max difference is ~0, the retrieval is provably correct.

Exact verification requires that the prematch run did NOT subsample the pool
(i.e. each speaker's/other pool <= --pool_cap), so neighbors are deterministic.
Run prematch with a big --pool_cap on a small set to check, e.g. --pool_cap 100000000.

Usage:
  python verify_prematch.py --orig logs/<exp>/filelist.txt \
      --prematched logs/<exp>/filelist_prematched_same.txt --pool same --k 4
"""

import argparse, os
import numpy as np


def load_fl(path):
    rows = []
    for line in open(path):
        x = line.strip().split("|")
        if len(x) >= 3:
            rows.append(x)
    return rows


def is_mute(r):
    return os.path.basename(r[1]).lower() == "mute.npy"


ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True, help="original filelist.txt")
ap.add_argument("--prematched", required=True, help="filelist_prematched_*.txt")
ap.add_argument("--pool", choices=["same", "other"], required=True)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--n_check", type=int, default=20, help="utterances to verify")
args = ap.parse_args()

orig = load_fl(args.orig)
pm = load_fl(args.prematched)
pm_by_wav = {r[0]: r for r in pm}

# --- structural checks ---
assert len(orig) == len(pm), f"row count changed: {len(orig)} vs {len(pm)}"
o_mute = sorted("|".join(r) for r in orig if is_mute(r))
p_mute = sorted("|".join(r) for r in pm if is_mute(r))
assert o_mute == p_mute, "MUTE ROWS CHANGED — bug!"
print(f"structure OK: {len(orig)} rows total, {len(o_mute)} mute rows identical")

orig_real = [r for r in orig if not is_mute(r)]
by_sid = {}
for r in orig_real:
    by_sid.setdefault(r[-1], []).append(r)

# --- numerical check: recompute each prematched frame with plain numpy ---
rng = np.random.default_rng(0)
sample = list(orig_real)
rng.shuffle(sample)

checked, maxdiff = 0, 0.0
for r in sample:
    if checked >= args.n_check:
        break
    wav, ofeat, sid = r[0], r[1], r[-1]
    q = np.load(ofeat).astype(np.float32)                     # this utt's own content
    got = np.load(pm_by_wav[wav][1]).astype(np.float32)       # what the script produced
    assert got.shape == q.shape, f"frame count changed for {wav}: {got.shape} vs {q.shape}"

    # oracle pool = exactly what the script is supposed to search
    if args.pool == "same":
        pool = [np.load(x[1]).astype(np.float32) for x in by_sid[sid] if x[0] != wav]
    else:  # other speakers
        pool = [np.load(x[1]).astype(np.float32)
                for s2, rs in by_sid.items() if s2 != sid for x in rs]
    if not pool:
        continue
    pool = np.concatenate(pool, 0)

    # brute-force exact L2 k-NN, average the k nearest (self already excluded by pool)
    d = ((q[:, None, :] - pool[None, :, :]) ** 2).sum(-1)     # [T, N]
    nn = np.argsort(d, axis=1)[:, :args.k]                    # k nearest per frame
    oracle = pool[nn].mean(1)

    diff = np.abs(oracle - got).max()
    maxdiff = max(maxdiff, diff)
    checked += 1

print(f"verified {checked} utterances (pool={args.pool}, k={args.k})")
print(f"MAX abs difference  (independent numpy  vs  script output) = {maxdiff:.2e}")
print("PASS: retrieval is exactly correct" if maxdiff < 1e-3 else "FAIL: mismatch — investigate")
