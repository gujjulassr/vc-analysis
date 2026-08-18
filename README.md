# vc-analysis

Rigorous evaluation & comparison toolkit for voice conversion systems.
Metrics: speaker similarity (SECS via Resemblyzer), source leakage, CER/WER
(ASR pseudo-reference via Whisper), UTMOS naturalness, paired Wilcoxon
significance tests, and worst-case transcript analysis.

## Install

```bash
pip install -r requirements.txt
```

## Workflow

### Step 0 — chunk long guided tracks into speech-only segments

Long tracks (e.g. 15 min with ~2 min of speech) must be cut into speech chunks
first: Whisper hallucinates on silence, UTMOS is meaningless on long silent
files, and speaker embeddings dilute.

```bash
python prepare_eval_chunks.py --in_dir raw_tracks --out_dir eval/source_wavs
```

Produces 2–20 s speech chunks named `<track>_<idx>.wav`.

### Step 1 — run inference on the chunks with each system

Convert `eval/source_wavs/*.wav` with each model checkpoint, keeping the same
filenames:

```
eval/
├── source_wavs/        source speech chunks
├── ref_wavs/           clean target-speaker reference clips (any names)
├── infer_out_sysA/     outputs of system A (same filenames as source)
└── infer_out_sysB/     outputs of system B
```

### Step 2 — analyze

```bash
python analysis_vc.py \
    --source eval/source_wavs \
    --ref eval/ref_wavs \
    --system sysA=eval/infer_out_sysA \
    --system sysB=eval/infer_out_sysB \
    --lang mr --whisper medium --out results
```

First `--system` is the baseline for the paired Wilcoxon tests.
`--lang`: Whisper language code (`mr` Marathi, `ta` Tamil, `hi` Hindi, ...).

## Outputs

- `results_per_utt.csv` — every chunk x system x metric (raw data)
- `results_summary.txt` — mean±std table, Wilcoxon p-values, 5 worst
  utterances per system by CER with source/output transcripts

## Metric notes

- **SECS_tgt** — cosine(output, target ref) with Resemblyzer. Use an embedder
  *different* from any used in training losses to avoid metric gaming.
- **SECS_src** — cosine(output, source) = leakage; lower is better.
- **gap** — SECS_tgt − SECS_src; bigger = cleaner conversion.
- **CER/WER** — ASR(source) is used as pseudo-reference for ASR(output);
  ASR errors affect all systems equally and the paired test absorbs them.
- **UTMOS** — naturalness 1–5 (auto-downloads via torch.hub on first run).
