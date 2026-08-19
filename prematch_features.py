#!/usr/bin/env python
"""
prematch_features.py — build kNN-VC / FragmentVC-style "prematched" training features.

For each utterance, replace every ContentVec frame with the mean of its k nearest
neighbors retrieved from a POOL. Training a VC model on these teaches it to render
audio from RETRIEVED features -- matching what happens at inference (FAISS retrieval).

--pool same   neighbors from the target speaker's OWN other utterances (kNN-VC).
              Content keeps the target's residual timbre -> weaker disentanglement,
              safer content. Matches inference at HIGH index_rate.
--pool other  neighbors from OTHER speakers (same language, since 1 lang/model).
              Content carries someone else's timbre -> the model MUST take identity
              from emb_g, forcing content-path disentanglement. Builds pseudo-parallel
              cross-speaker pairs (foreign content -> target audio). Riskier content
              (cross-speaker NN is noisier). Matches inference at LOW index_rate.

Prior art to cite: kNN-VC (Interspeech 2023), FragmentVC (ICASSP 2021).
The 'other' pool + sweeping same-vs-other as an axis is the novel bit.

Usage:
  python prematch_features.py --filelist logs/<exp>/filelist.txt --k 4 --pool same
  python prematch_features.py --filelist logs/<exp>/filelist.txt --k 4 --pool other
Writes:
  <exp>/feature_prematched_<pool>/<sid>_<idx>.npy
  <exp>/filelist_prematched_<pool>.txt
Then train a NEW model whose training filelist is filelist_prematched_<pool>.txt.
"""

import argparse, os

import numpy as np
import faiss

ap = argparse.ArgumentParser()
ap.add_argument("--filelist", required=True, help="Applio filelist.txt")
ap.add_argument("--out_dir", default=None,
                help="default: <filelist dir>/feature_prematched_<pool>")
ap.add_argument("--k", type=int, default=4, help="neighbors to average per frame")
ap.add_argument("--pool", choices=["same", "other"], default="same",
                help="'same' = target speaker's other utts (kNN-VC); "
                     "'other' = OTHER speakers -> forces content-path disentanglement")
ap.add_argument("--pool_cap", type=int, default=100000,
                help="max pool frames for --pool other (subsampled per speaker if larger)")
args = ap.parse_args()
rng = np.random.default_rng(0)

exp_dir = os.path.dirname(os.path.abspath(args.filelist))
out_dir = args.out_dir or os.path.join(exp_dir, f"feature_prematched_{args.pool}")
os.makedirs(out_dir, exist_ok=True)

# filelist format:  wav | feat | f0 | f0nsf | sid   (sid = last field, feat = 2nd)
# Mute/silence rows (feat basename == mute.npy) are passed through UNCHANGED --
# never prematch the silence token or use it in a retrieval pool.
rows, by_sid, passthrough = [], {}, []
with open(args.filelist) as f:
    for line in f:
        p = line.strip().split("|")
        if len(p) < 3:
            continue
        if os.path.basename(p[1]).lower() == "mute.npy":
            passthrough.append("|".join(p))          # keep as-is, don't retrieve
            continue
        rows.append(p)
        by_sid.setdefault(p[-1], []).append(len(rows) - 1)
print(f"{len(rows)} utterances, {len(by_sid)} speakers, "
      f"{len(passthrough)} mute rows passed through, pool={args.pool}")

# preload all features (needed so 'other' can pool across speakers)
feats = {}   # row_idx -> [T, dim]
for idxs in by_sid.values():
    for i in idxs:
        try:
            feats[i] = np.load(rows[i][1]).astype(np.float32)
        except Exception as e:
            print(f"  skip (load fail) {rows[i][1]}: {e}")

new_lines = []


def save(sid, i, pm):
    outp = os.path.join(out_dir, f"{sid}_{i}.npy")
    np.save(outp, pm.astype(np.float32))
    p = list(rows[i]); p[1] = outp
    new_lines.append("|".join(p))


def make_index(pool):
    ix = faiss.IndexFlatL2(pool.shape[1])
    ix.add(pool)
    return ix


def other_pool(sid):
    """All OTHER speakers' frames, capped ~pool_cap (subsampled per speaker)."""
    others = [s2 for s2 in by_sid if s2 != sid]
    if not others:
        return None
    per = max(1, args.pool_cap // len(others))
    chunks = []
    for s2 in others:
        frames = [feats[j] for j in by_sid[s2] if j in feats]
        if not frames:
            continue
        a = np.concatenate(frames, 0)
        if len(a) > per:
            a = a[rng.choice(len(a), per, replace=False)]
        chunks.append(a)
    return np.concatenate(chunks, 0) if chunks else None


for sid, idxs in by_sid.items():
    if args.pool == "other":
        pool = other_pool(sid)
        if pool is None:                       # only one speaker in the whole set
            for i in idxs:
                if i in feats:
                    save(sid, i, feats[i])
            continue
        index = make_index(pool)
        for i in idxs:
            if i not in feats:
                continue
            _, I = index.search(feats[i], min(args.k, len(pool)))
            save(sid, i, pool[I].mean(1))
    else:                                       # same speaker's other utts
        for i in idxs:
            if i not in feats:
                continue
            q = feats[i]
            others = [feats[j] for j in idxs if j != i and j in feats]
            if not others:                      # speaker has a single utterance
                save(sid, i, q)
                continue
            pool = np.concatenate(others, 0)
            index = make_index(pool)
            _, I = index.search(q, min(args.k, len(pool)))
            save(sid, i, pool[I].mean(1))
    print(f"sid {sid}: {len(idxs)} utts prematched ({args.pool})")

out_fl = os.path.join(exp_dir, f"filelist_prematched_{args.pool}.txt")
with open(out_fl, "w") as f:
    f.write("\n".join(new_lines + passthrough) + "\n")   # mute rows kept unchanged
print(f"\nwrote {out_fl}\n      {out_dir}/")
print(f"{len(new_lines)} prematched + {len(passthrough)} mute rows = "
      f"{len(new_lines) + len(passthrough)} total")
print("Next: set PREMATCH in train.py (line ~144) and run "
      f"PREMATCH={args.pool} AUX=1 bash train.sh")
