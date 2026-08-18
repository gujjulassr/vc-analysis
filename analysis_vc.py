#!/usr/bin/env python
"""
analysis_vc.py — rigorous VC evaluation & multi-system comparison (paper-grade).

Compares any number of systems (e.g. spk+content losses vs v2 attention) on the
SAME source utterances, with:
  SECS_tgt   speaker similarity to target   (Resemblyzer — independent of the
             ECAPA used in training, so no metric gaming)
  SECS_src   similarity to source = LEAKAGE (lower is better)
  gap        SECS_tgt - SECS_src            (bigger = cleaner conversion)
  CER / WER  pronunciation: ASR(source) as pseudo-reference vs ASR(output)
  UTMOS      naturalness 1-5
  Wilcoxon   paired significance test between system 1 and each other system
  Worst-5    utterances by CER, with transcripts printed -> WHERE it breaks

Install once:
  pip install openai-whisper jiwer resemblyzer scipy librosa soundfile

Usage (ONE LANGUAGE PER RUN; --lang is the Whisper code: hi, mr, ta, ...):
  python analysis_vc.py \
      --source <eval_LANGUAGE>/source_wavs \
      --ref <eval_LANGUAGE>/ref_wavs \
      --system converted=<eval_LANGUAGE>/infer_out_converted \
      --lang <code> --whisper medium --out results_LANGUAGE

Folder rules:
  source/<name>.wav               the source utterances
  each system dir/<name>.wav      SAME filenames = converted outputs
  ref/<name>.wav if it exists     per-utterance target reference,
  else                            ALL wavs in ref/ are averaged into one
                                  target-speaker centroid.
"""

import argparse, csv, glob, os, re, unicodedata

import numpy as np
import torch
import librosa
from scipy import stats

# ---------------- args ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--system", action="append", required=True,
                help="name=dir  (repeat per system; first = baseline)")
ap.add_argument("--lang", default="mr", help="ASR language code (mr/ta/hi)")
ap.add_argument("--whisper", default="medium", help="whisper model size")
ap.add_argument("--out", default="results", help="output file prefix")
ap.add_argument("--transcribe_refs", action="store_true",
                help="also ASR the reference wavs into the transcripts file "
                     "(slow: refs are long files)")
args = ap.parse_args()

systems = []
for s in args.system:
    name, d = s.split("=", 1)
    systems.append((name, d))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- models ----------------
print("loading whisper ...")
import whisper
asr = whisper.load_model(args.whisper, device=device)

print("loading UTMOS ...")
utmos = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong",
                       trust_repo=True).to(device).eval()

print("loading Resemblyzer ...")
from resemblyzer import VoiceEncoder, preprocess_wav
spk = VoiceEncoder(device)

# ---------------- helpers ----------------
_punct = re.compile(r"[^\w\s]", re.UNICODE)

def norm_text(t):
    t = unicodedata.normalize("NFC", t)
    t = _punct.sub(" ", t.lower())
    return " ".join(t.split())

_asr_cache = {}
def transcribe(path):
    if path not in _asr_cache:
        r = asr.transcribe(path, language=args.lang, fp16=(device == "cuda"))
        _asr_cache[path] = norm_text(r["text"])
    return _asr_cache[path]

def mos(path):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    with torch.no_grad():
        return utmos(torch.from_numpy(wav).unsqueeze(0).to(device), 16000).item()

_emb_cache = {}
def emb(path):
    if path not in _emb_cache:
        _emb_cache[path] = spk.embed_utterance(preprocess_wav(path))
    return _emb_cache[path]

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

import jiwer

# ---------------- target references ----------------
# lookup order per chunk <base>_<idx>.wav:
#   1. exact ref/<name>.wav
#   2. per-track refs ref/<base>__ref*.wav  (from prepare_guided_eval.py)
#   3. global centroid of everything in ref/
ref_wavs = sorted(glob.glob(os.path.join(args.ref, "*.wav")))
assert ref_wavs, f"no wavs in {args.ref}"
centroid = np.mean([emb(p) for p in ref_wavs], axis=0)

_tgt_cache = {}
def target_emb(name):
    exact = os.path.join(args.ref, name)
    if os.path.exists(exact):
        return emb(exact)
    base = re.sub(r"_\d+\.wav$", "", name)
    if base not in _tgt_cache:
        cand = sorted(glob.glob(os.path.join(args.ref, base + "__*.wav")))
        _tgt_cache[base] = (np.mean([emb(p) for p in cand], axis=0)
                            if cand else centroid)
    return _tgt_cache[base]

# ---------------- matched file list ----------------
names = set(os.path.basename(p) for p in glob.glob(os.path.join(args.source, "*.wav")))
for _, d in systems:
    names &= set(os.path.basename(p) for p in glob.glob(os.path.join(d, "*.wav")))
names = sorted(names)
assert names, "no matching filenames across source + all system dirs"
print(f"{len(names)} matched utterances | systems: {[n for n, _ in systems]}")

# ---------------- evaluate ----------------
rows = []           # dicts: name, system, secs_tgt, secs_src, gap, cer, wer, utmos
texts = {}          # (system, name) -> hyp transcript ; ("src", name) -> source
for i, name in enumerate(names):
    srcp = os.path.join(args.source, name)
    src_text = transcribe(srcp)
    texts[("src", name)] = src_text
    src_e = emb(srcp)
    tgt_e = target_emb(name)

    for sysname, d in systems:
        outp = os.path.join(d, name)
        hyp = transcribe(outp)
        texts[(sysname, name)] = hyp
        e = emb(outp)
        cer = jiwer.cer(src_text, hyp) if src_text else float("nan")
        wer = jiwer.wer(src_text, hyp) if src_text else float("nan")
        rows.append(dict(
            name=name, system=sysname,
            secs_tgt=cos(e, tgt_e), secs_src=cos(e, src_e),
            gap=cos(e, tgt_e) - cos(e, src_e),
            cer=cer, wer=wer, utmos=mos(outp),
        ))
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(names)} done")

# ---------------- per-utterance CSV ----------------
csv_path = f"{args.out}_per_utt.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {csv_path}")

# ---------------- summary + significance ----------------
METRICS = ["secs_tgt", "secs_src", "gap", "cer", "wer", "utmos"]

def col(sysname, metric):
    return np.array([r[metric] for r in rows if r["system"] == sysname])

lines = []
lines.append(f"{'system':12s} " + " ".join(f"{m:>14s}" for m in METRICS))
for sysname, _ in systems:
    vals = [col(sysname, m) for m in METRICS]
    lines.append(f"{sysname:12s} " + " ".join(
        f"{v.mean():7.3f}±{v.std():5.3f}" for v in vals))

base = systems[0][0]
for sysname, _ in systems[1:]:
    lines.append(f"\nWilcoxon paired: {base} vs {sysname}  (p<0.05 = significant)")
    for m in METRICS:
        a, b = col(base, m), col(sysname, m)
        try:
            p = stats.wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
        except ValueError:
            p = float("nan")
        lines.append(f"  {m:10s} {base}={a.mean():.3f} {sysname}={b.mean():.3f} "
                     f"diff={b.mean() - a.mean():+.4f}  p={p:.4f}")

# ---------------- worst cases: WHERE pronunciation breaks ----------------
for sysname, _ in systems:
    lines.append(f"\n===== {sysname}: 5 worst utterances by CER =====")
    worst = sorted((r for r in rows if r["system"] == sysname),
                   key=lambda r: -r["cer"])[:5]
    for r in worst:
        lines.append(f"[{r['name']}] CER={r['cer']:.3f} SECS_tgt={r['secs_tgt']:.3f}")
        lines.append(f"  SRC: {texts[('src', r['name'])]}")
        lines.append(f"  OUT: {texts[(sysname, r['name'])]}")

report = "\n".join(lines)
print("\n" + report)
with open(f"{args.out}_summary.txt", "w") as f:
    f.write(report + "\n")
print(f"\nwrote {args.out}_summary.txt")

# ---------------- averages CSV (one row per system) ----------------
apath = f"{args.out}_averages.csv"
with open(apath, "w", newline="") as f:
    w = csv.writer(f)
    header = ["system", "n"]
    for m in METRICS:
        header += [f"{m}_mean", f"{m}_std"]
    w.writerow(header)
    for sysname, _ in systems:
        row = [sysname, len(col(sysname, METRICS[0]))]
        for m in METRICS:
            v = col(sysname, m)
            row += [f"{np.nanmean(v):.4f}", f"{np.nanstd(v):.4f}"]
        w.writerow(row)
print(f"wrote {apath}")

# ---------------- full transcripts dump: SRC vs each system (vs REF) ----------------
tpath = f"{args.out}_transcripts.txt"
with open(tpath, "w") as f:
    cur = None
    for name in names:
        base = re.sub(r"_\d+\.wav$", "", name)
        if base != cur:
            f.write(f"\n========== {base} ==========\n")
            if args.transcribe_refs:
                for rp in sorted(glob.glob(os.path.join(args.ref, base + "__*.wav"))):
                    print(f"transcribing ref {os.path.basename(rp)} ...")
                    f.write(f"REF [{os.path.basename(rp)}]:\n  {transcribe(rp)}\n")
            cur = base
        f.write(f"\n[{name}]\n")
        f.write(f"  SRC : {texts[('src', name)]}\n")
        for sysname, _ in systems:
            f.write(f"  {sysname.upper():4s}: {texts[(sysname, name)]}\n")
print(f"wrote {tpath}")
