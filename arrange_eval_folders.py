#!/usr/bin/env python
"""
arrange_eval_folders.py — ONLY arranges audios into the eval folder layout.
No cutting, no processing. Chunking is done afterwards by prepare_eval_chunks.py.

Builds:
  eval_dir/
    source_full/          copies of the source guided tracks (FILE_SID keys)
    out_full_<system>/    copies of each system's full-length outputs
    ref_wavs/             per-track target references named <base>__refN.wav,
                          resolved via FILE_SID -> sid -> speaker_map.txt ->
                          voicebank/<artist>/ (the N LONGEST wav files)

ONE LANGUAGE PER RUN — the language lives in the arguments, not in this script.
Run it once per language with that language's outputs dir, voicebank,
speaker_map and mapping file.

Usage (example for one language):
  python arrange_eval_folders.py \
      --tracks_dir <guided_tracks_dir> \
      --outputs converted=<dir with the converted full-length outputs> \
      --voicebank <voicebank/LANGUAGE> \
      --speaker_map <logs/MODEL/speaker_map.txt> \
      --file_sid <language_map.txt: one 'base=sid' per line> \
      --eval_dir <eval_LANGUAGE>
"""

import argparse, glob, os, shutil

import soundfile as sf

# track base name -> sid used at conversion time.
# Prefer --file_sid <file>; this inline dict is only a fallback/example.
FILE_SID = {
    # "Shaun_01": 30,
}

ap = argparse.ArgumentParser()
ap.add_argument("--tracks_dir", required=True)
ap.add_argument("--outputs", action="append", required=True, help="name=dir (repeat)")
ap.add_argument("--voicebank", required=True)
ap.add_argument("--speaker_map", required=True)
ap.add_argument("--eval_dir", required=True)
ap.add_argument("--refs_per_spk", type=int, default=5)
ap.add_argument("--file_sid", default=None,
                help="optional mapping file, one 'base=sid' per line; "
                     "overrides the FILE_SID dict above")
args = ap.parse_args()

systems = [s.split("=", 1) for s in args.outputs]

if args.file_sid:
    FILE_SID = {}
    with open(args.file_sid) as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                base, sid = line.rsplit("=", 1)
                FILE_SID[base.strip()] = int(sid.strip())
    print(f"file_sid: {len(FILE_SID)} tracks from {args.file_sid}")

# sid -> artist folder name  ("30  Artist-9" -> {30: "Artist-9"})
sid2artist = {}
with open(args.speaker_map) as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            sid2artist[int(parts[0])] = parts[1].strip()
print(f"speaker_map: {len(sid2artist)} sids")

src_dir = os.path.join(args.eval_dir, "source_full")
ref_dir = os.path.join(args.eval_dir, "ref_wavs")
os.makedirs(src_dir, exist_ok=True)
os.makedirs(ref_dir, exist_ok=True)
out_dirs = {}
for name, _ in systems:
    out_dirs[name] = os.path.join(args.eval_dir, f"out_full_{name}")
    os.makedirs(out_dirs[name], exist_ok=True)

def longest_wavs(folder, n):
    """n longest wav files (by duration) under folder, recursive."""
    wavs = glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True)
    with_dur = []
    for p in wavs:
        try:
            with_dur.append((sf.info(p).duration, p))
        except Exception:
            pass
    with_dur.sort(key=lambda x: -x[0])
    return with_dur[:n]

copied = 0
for base, sid in FILE_SID.items():
    srcp = os.path.join(args.tracks_dir, base + ".wav")
    if not os.path.exists(srcp):
        print(f"[skip] missing source: {srcp}")
        continue
    artist = sid2artist.get(sid)
    if artist is None:
        print(f"[skip] {base}: sid {sid} not in speaker_map")
        continue

    ok = True
    for name, d in systems:
        outp = os.path.join(d, base + ".wav")
        if not os.path.exists(outp):
            print(f"[skip] {base}: missing output {outp}")
            ok = False
    if not ok:
        continue

    shutil.copy2(srcp, os.path.join(src_dir, base + ".wav"))
    for name, d in systems:
        shutil.copy2(os.path.join(d, base + ".wav"),
                     os.path.join(out_dirs[name], base + ".wav"))

    refs = longest_wavs(os.path.join(args.voicebank, artist), args.refs_per_spk)
    if not refs:
        print(f"[warn] {base}: no wavs for artist '{artist}'")
    for j, (dur, p) in enumerate(refs):
        shutil.copy2(p, os.path.join(ref_dir, f"{base}__ref{j}.wav"))

    print(f"{base}: sid={sid} -> '{artist}' | {len(refs)} refs "
          f"(longest {refs[0][0]:.1f}s)" if refs else f"{base}: sid={sid} -> '{artist}'")
    copied += 1

print(f"\nDONE: {copied} tracks arranged in {args.eval_dir}")
print("Next: chunk with prepare_eval_chunks.py (use --aligned for the outputs),")
print("then run analysis_vc.py.")
