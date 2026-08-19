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

import argparse, os, time

import numpy as np
import faiss

try:
    from tqdm import tqdm
except Exception:                        # tqdm optional; falls back to per-speaker prints
    tqdm = None

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
ap.add_argument("--cpu", action="store_true", help="force CPU even if faiss-gpu is present")
args = ap.parse_args()
rng = np.random.default_rng(0)

faiss.omp_set_num_threads(os.cpu_count() or 4)     # use all cores for CPU search

# Use the GPU for brute-force search if faiss-gpu is installed (10-50x faster).
USE_GPU, _GPU_RES = False, None
if not args.cpu:
    try:
        _GPU_RES = faiss.StandardGpuResources()
        USE_GPU = True
        print("FAISS: using GPU")
    except Exception:
        print(f"FAISS: GPU not available -> CPU x{os.cpu_count()} threads "
              "(pip install faiss-gpu for a further speedup)")

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
bar = tqdm(total=len(rows), desc=f"prematch({args.pool})", unit="utt") if tqdm else None


def save(sid, i, pm):
    outp = os.path.join(out_dir, f"{sid}_{i}.npy")
    np.save(outp, pm.astype(np.float32))
    p = list(rows[i]); p[1] = outp
    new_lines.append("|".join(p))
    if bar:
        bar.update(1)


def make_index(pool):
    ix = faiss.IndexFlatL2(pool.shape[1])
    if USE_GPU:
        ix = faiss.index_cpu_to_gpu(_GPU_RES, 0, ix)
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


t0 = time.time()
n_spk = len(by_sid)
for si, (sid, idxs) in enumerate(by_sid.items(), 1):
    present = [i for i in idxs if i in feats]
    if args.pool == "other":
        pool = other_pool(sid)
        if pool is None:                       # only one speaker in the whole set
            for i in present:
                save(sid, i, feats[i])
            continue
        index = make_index(pool)
        for i in present:
            _, I = index.search(feats[i], min(args.k, len(pool)))
            save(sid, i, pool[I].mean(1))      # no self-exclusion: pool is other speakers
    else:                                       # same speaker: ONE index over own frames
        # concat the speaker's frames, remembering each utterance's [start,end) range
        mats, ranges, order = [], {}, []
        start = 0
        for i in present:
            a = feats[i]
            ranges[i] = (start, start + len(a))
            mats.append(a); order.append(i)
            start += len(a)
        if len(order) <= 1:                     # single utterance -> can't prematch
            for i in order:
                save(sid, i, feats[i])
            continue
        allf = np.concatenate(mats, 0)
        index = make_index(allf)
        for i in order:
            s, e = ranges[i]
            q = feats[i]
            kk = min(args.k + (e - s), len(allf), 2048)    # drop self; 2048 = GPU search cap
            _, I = index.search(q, kk)                      # [T, kk]
            selfmask = (I >= s) & (I < e)                   # True where neighbor is self
            # stable-sort valid(False) before self(True), keep NN order -> first k are valid
            keep = np.take_along_axis(I, np.argsort(selfmask, axis=1, kind="stable"), 1)
            sel = keep[:, :args.k]                          # [T, k] non-self neighbors
            save(sid, i, allf[sel].mean(1))                 # [T,k,dim] -> [T,dim]
    if USE_GPU:
        del index                               # free GPU memory before next speaker
    if bar is None:                             # no tqdm -> coarse per-speaker log
        print(f"[{si}/{n_spk}] sid {sid}: {len(present)} utts prematched "
              f"({args.pool}, {time.time()-t0:.0f}s elapsed)", flush=True)
if bar:
    bar.close()

out_fl = os.path.join(exp_dir, f"filelist_prematched_{args.pool}.txt")
with open(out_fl, "w") as f:
    f.write("\n".join(new_lines + passthrough) + "\n")   # mute rows kept unchanged
print(f"\nwrote {out_fl}\n      {out_dir}/")
print(f"{len(new_lines)} prematched + {len(passthrough)} mute rows = "
      f"{len(new_lines) + len(passthrough)} total")
print("Next: set PREMATCH in train.py (line ~144) and run "
      f"PREMATCH={args.pool} AUX=1 bash train.sh")
