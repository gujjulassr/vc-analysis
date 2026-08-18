#!/usr/bin/env python
"""
prepare_eval_chunks.py — cut long guided tracks into speech-only chunks for evaluation.

Long track (15 min, ~2 min speech) -> N short speech chunks (2-20 s each), named
<track>_<idx>.wav. Run inference on THESE chunks with each model, then run
analysis_vc.py on the chunk folders.

Why chunks first: Whisper hallucinates on silence, UTMOS is meaningless on long
silent files, and speaker embeddings dilute. Chunks fix all three, and give many
paired samples per track (stronger Wilcoxon).

Install:  pip install silero-vad soundfile librosa
Usage:    python prepare_eval_chunks.py --in_dir raw_tracks --out_dir source_wavs

Aligned outputs (converted files time-aligned with the sources): cut them at the
SAME source timestamps so chunk filenames pair up exactly. ONE LANGUAGE PER RUN:
  python prepare_eval_chunks.py \
      --in_dir <eval_LANGUAGE>/source_full --out_dir <eval_LANGUAGE>/source_wavs \
      --aligned <eval_LANGUAGE>/out_full_converted=<eval_LANGUAGE>/infer_out_converted
"""

import argparse, glob, os

import numpy as np
import soundfile as sf
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True, help="folder of long source tracks")
ap.add_argument("--out_dir", required=True, help="output folder for speech chunks")
ap.add_argument("--min_len", type=float, default=2.0, help="drop chunks shorter (s)")
ap.add_argument("--max_len", type=float, default=20.0, help="split chunks longer (s)")
ap.add_argument("--merge_gap", type=float, default=0.6, help="merge if gap < (s)")
ap.add_argument("--aligned", action="append", default=[],
                help="indir=outdir: full-length dirs time-aligned with the source; "
                     "cut at the SOURCE's timestamps (repeatable)")
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
aligned = []
for s in args.aligned:
    indir, outdir = s.split("=", 1)
    os.makedirs(outdir, exist_ok=True)
    aligned.append((indir, outdir))
vad = load_silero_vad()
total = 0

for path in sorted(glob.glob(os.path.join(args.in_dir, "*.wav"))):
    base = os.path.splitext(os.path.basename(path))[0]
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(1)
    audio = audio.astype(np.float32)

    w16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    ts = get_speech_timestamps(
        torch.from_numpy(w16).float(), vad, sampling_rate=16000,
        threshold=0.35, min_speech_duration_ms=250,
        min_silence_duration_ms=200, speech_pad_ms=150)
    sc = sr / 16000
    segs = [(int(t["start"] * sc), int(t["end"] * sc)) for t in ts]

    # merge segments separated by short gaps
    merged = []
    for a, b in segs:
        if merged and a - merged[-1][1] <= args.merge_gap * sr:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    # enforce min/max chunk length
    chunks = []
    for a, b in merged:
        dur = (b - a) / sr
        if dur < args.min_len:
            continue
        step = int(args.max_len * sr)
        while b - a > step:                    # split over-long chunks
            chunks.append((a, a + step))
            a += step
        chunks.append((a, b))

    for i, (a, b) in enumerate(chunks):
        out = os.path.join(args.out_dir, f"{base}_{i:03d}.wav")
        sf.write(out, audio[a:b], sr, subtype="PCM_16")

    # cut aligned full-length outputs at the SAME source timestamps
    for indir, outdir in aligned:
        ap_ = os.path.join(indir, f"{base}.wav")
        if not os.path.exists(ap_):
            print(f"  [warn] no aligned file {ap_}")
            continue
        oa, osr = sf.read(ap_)
        if oa.ndim > 1:
            oa = oa.mean(1)
        oa = oa.astype(np.float32)
        if abs(len(oa) / osr - len(audio) / sr) > 0.5:
            print(f"  [warn] {base}: aligned duration differs "
                  f"{abs(len(oa)/osr - len(audio)/sr):.2f}s ({indir})")
        for i, (a, b) in enumerate(chunks):
            a2, b2 = int(a / sr * osr), int(b / sr * osr)
            sf.write(os.path.join(outdir, f"{base}_{i:03d}.wav"),
                     oa[a2:b2], osr, subtype="PCM_16")
    print(f"{base}: {len(chunks)} chunks "
          f"({sum(b - a for a, b in chunks) / sr:.1f}s speech of {len(audio) / sr:.1f}s)")
    total += len(chunks)

print(f"\ntotal: {total} chunks in {args.out_dir}")
