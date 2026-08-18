# vc-analysis

Rigorous evaluation & comparison toolkit for voice conversion systems.
Metrics: speaker similarity (SECS via Resemblyzer), source leakage, CER/WER
(ASR pseudo-reference via Whisper, Indic-safe text normalization), UTMOS
naturalness, paired Wilcoxon significance tests, full transcript dumps and
worst-case analysis.

**Rule: ONE LANGUAGE PER RUN.** The language lives in your arguments
(`eval_hindi/`, `--lang hi`, `hindi_map.txt`) — never inside the scripts.

## Install

```bash
pip install -r requirements.txt
```

## Workflow A — guided tracks (long sources + aligned full-length outputs)

For long tracks (e.g. 15 min with ~2 min of speech) that were already
converted full-length with timing preserved (output aligned with source).

### A0. Mapping file `<language>_map.txt`

One `trackname=sid` per line — the sid each track was converted with
(copy from your inference script's FILE_SID dict):

```
Shaun_01=30
Aoki_01=24
```

### A1. Arrange — collect everything into one eval folder

```bash
python arrange_eval_folders.py \
    --tracks_dir <dir with source tracks> \
    --outputs converted=<dir with converted full-length outputs> \
    --voicebank <voicebank/LANGUAGE> \
    --speaker_map <logs/MODEL/speaker_map.txt> \
    --file_sid <language>_map.txt \
    --eval_dir eval_<language>
```

Copies (creates, reads-only from inputs):
- `source_full/` — the mapped source tracks
- `out_full_converted/` — the converted outputs
- `ref_wavs/<base>__refN.wav` — per-track target references, resolved via
  FILE_SID -> sid -> speaker_map.txt -> voicebank/<artist>/ (N longest files)

### A2. Chunk — source and outputs cut at the SAME timestamps

```bash
python prepare_eval_chunks.py \
    --in_dir eval_<language>/source_full \
    --out_dir eval_<language>/source_wavs \
    --aligned eval_<language>/out_full_converted=eval_<language>/infer_out_converted \
    --min_len 3.0
```

VAD runs on the source; outputs are cut at identical timestamps, so
`source_wavs/` and `infer_out_converted/` contain the same chunk names =
the same moments. Verify: both folders must have equal file counts.
(The full-length staging folders can be deleted after this step.)

### A3. Analyze

```bash
python analysis_vc.py \
    --source eval_<language>/source_wavs \
    --ref eval_<language>/ref_wavs \
    --system <label>=eval_<language>/infer_out_converted \
    --lang <hi|mr|ta|...> --whisper medium --out results_<language>
```

## Workflow B — generic (short sources, infer per chunk)

1. Chunk long sources: `prepare_eval_chunks.py --in_dir raw --out_dir source_wavs`
2. Run inference on the chunks with each model (same filenames per system dir)
3. `analysis_vc.py` with one `--system name=dir` per model — with 2+ systems the
   first is the baseline and paired Wilcoxon tests are reported.

## Outputs

- `<out>_per_utt.csv` — every chunk x system x metric (raw data)
- `<out>_averages.csv` — one row per system, mean/std of every metric
- `<out>_summary.txt` — aggregate table (nan-safe), Wilcoxon tests, worst-5
  utterances by CER with transcripts
- `<out>_transcripts.txt` — full SRC-vs-system transcripts per chunk
  (add `--transcribe_refs` to include REF transcripts; slow on long refs)

## Metric notes

- **SECS_tgt** — cosine(output, target ref) with Resemblyzer. Use an embedder
  *different* from any used in training losses to avoid metric gaming.
- **SECS_src** — cosine(output, source) = leakage; lower is better.
- **gap** — SECS_tgt − SECS_src; bigger = cleaner conversion.
- **CER/WER** — ASR(source) is the pseudo-reference for ASR(output); ASR errors
  hit all systems equally and the paired test absorbs them. Normalization keeps
  Unicode combining marks (Indic matras survive). Chunks with empty source ASR
  are excluded (count reported in the summary).
- **UTMOS** — naturalness 1–5 (auto-downloads via torch.hub on first run).
