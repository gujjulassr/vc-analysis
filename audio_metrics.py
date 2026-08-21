#!/usr/bin/env python
"""
audio_metrics.py — ONE ROW PER AUDIO: speaker similarity on voiced speech only.

NO chunking involved. Each full-length audio is scanned with VAD, all voiced
samples are concatenated, and ONE speaker embedding is computed on that voiced
audio. Silences cannot affect the score, and even a track with only a few
seconds of speech gets a row (chunk minimum-length rules do not exist here).

This tool does SIMILARITY ONLY (simple by design). CER/UTMOS need short
segments, so they stay in the chunk-based analysis_vc.py flow.

Usage (full-length files, same filenames across dirs):
  python audio_metrics.py \
      --system base=out_full_converted \
      --system pm_same=out_full_pm \
      --ref ref_wavs \
      [--source source_full]        # adds SECS_src (leak) + gap
      [--out audio_metrics.csv]

Reference lookup per audio <name>.wav:
  ref/<name>__*.wav (averaged)  else  ALL wavs in ref/ averaged (centroid).

Output CSV columns: audio, system, speech_sec, secs_tgt[, secs_src, gap]
"""

import argparse, csv, glob, os

import numpy as np
import soundfile as sf
import librosa
import torch
from silero_vad import load_silero_vad, get_speech_timestamps
from resemblyzer import VoiceEncoder, preprocess_wav

ap = argparse.ArgumentParser()
ap.add_argument("--system", action="append", required=True,
                help="name=dir of full-length converted audios (repeatable)")
ap.add_argument("--ref", required=True, help="target reference wavs")
ap.add_argument("--source", default=None,
                help="full-length source audios -> adds SECS_src (leak) + gap")
ap.add_argument("--out", default="audio_metrics.csv")
ap.add_argument("--vad_threshold", type=float, default=0.35,
                help="lower (e.g. 0.2) rescues quiet stems")
ap.add_argument("--min_speech", type=float, default=1.0,
                help="below this many voiced seconds the row is marked NaN")
args = ap.parse_args()

systems = []
for s in args.system:
    if "=" not in s:
        ap.error(f"--system needs name=dir (e.g. --system base=out_full_converted), got: {s!r}")
    systems.append(tuple(s.split("=", 1)))
device = "cuda" if torch.cuda.is_available() else "cpu"
vad = load_silero_vad()
enc = VoiceEncoder(device)


def voiced16k(path):
    """All voiced audio of a file, concatenated, at 16 kHz. Returns (wav, seconds)."""
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(1)
    w = librosa.resample(a.astype(np.float32), orig_sr=sr, target_sr=16000)
    ts = get_speech_timestamps(
        torch.from_numpy(w).float(), vad, sampling_rate=16000,
        threshold=args.vad_threshold, min_speech_duration_ms=250,
        min_silence_duration_ms=200, speech_pad_ms=100)
    if not ts:
        return None, 0.0
    v = np.concatenate([w[t["start"]:t["end"]] for t in ts])
    return v, len(v) / 16000.0


_vemb_cache = {}
def voiced_emb(path):
    """Speaker embedding of a file's concatenated voiced audio (None if silent)."""
    if path not in _vemb_cache:
        v, sec = voiced16k(path)
        e = (enc.embed_utterance(preprocess_wav(v, source_sr=16000))
             if v is not None and sec >= args.min_speech else None)
        _vemb_cache[path] = (e, sec)
    return _vemb_cache[path]


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---- reference embeddings (refs are speech-dense: embed whole files) ----
ref_all = sorted(glob.glob(os.path.join(args.ref, "*.wav")))
assert ref_all, f"no wavs in {args.ref}"
_remb = {p: enc.embed_utterance(preprocess_wav(p)) for p in ref_all}
centroid = np.mean(list(_remb.values()), axis=0)

def target_emb(name):
    base = os.path.splitext(name)[0]
    cand = sorted(glob.glob(os.path.join(args.ref, base + "__*.wav")))
    return np.mean([_remb.get(p, enc.embed_utterance(preprocess_wav(p)))
                    for p in cand], axis=0) if cand else centroid


# ---- audios present in every system dir ----
names = None
for _, d in systems:
    got = set(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.wav")))
    names = got if names is None else names & got
names = sorted(names or [])
assert names, "no matching filenames across system dirs"

rows = []
for name in names:
    tgt = target_emb(name)
    src_e = None
    if args.source:
        sp = os.path.join(args.source, name)
        if os.path.exists(sp):
            src_e, _ = voiced_emb(sp)
    # baseline: how similar the SOURCE and TARGET voices are to each other.
    # secs_src is only "leak" to the extent it EXCEEDS this number.
    src_tgt = round(cos(src_e, tgt), 4) if src_e is not None else float("nan")
    for sysname, d in systems:
        e, sec = voiced_emb(os.path.join(d, name))
        row = {"audio": name, "system": sysname, "speech_sec": round(sec, 1),
               "secs_tgt": round(cos(e, tgt), 4) if e is not None else float("nan")}
        if args.source:
            has = e is not None and src_e is not None
            row["secs_src"] = round(cos(e, src_e), 4) if has else float("nan")
            row["gap"] = round(row["secs_tgt"] - row["secs_src"], 4) if has else float("nan")
            row["src_tgt_baseline"] = src_tgt
            row["leak_excess"] = round(row["secs_src"] - src_tgt, 4) if has else float("nan")
        rows.append(row)
        note = "" if e is not None else "   <- too little voiced audio, similarity skipped"
        print(f"{name:35s} {sysname:12s} speech={sec:7.1f}s  " +
              "  ".join(f"{k}={row[k]}" for k in row if k not in ("audio", "system", "speech_sec"))
              + note)

with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {args.out}  ({len(names)} audios x {len(systems)} systems)")

# per-system means over audios with enough speech
print("\nmeans over audios (rows with enough voiced speech):")
for sysname, _ in systems:
    vals = [r for r in rows if r["system"] == sysname and not np.isnan(r["secs_tgt"])]
    if not vals:
        print(f"  {sysname:12s} (no valid audios)")
        continue
    msg = f"  {sysname:12s} n={len(vals):2d}  secs_tgt={np.mean([r['secs_tgt'] for r in vals]):.4f}"
    if args.source and not all(np.isnan(r.get("secs_src", np.nan)) for r in vals):
        ok = [r for r in vals if not np.isnan(r["secs_src"])]
        msg += (f"  secs_src={np.mean([r['secs_src'] for r in ok]):.4f}"
                f"  gap={np.mean([r['gap'] for r in ok]):.4f}")
    print(msg)
