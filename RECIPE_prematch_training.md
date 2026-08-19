# Recipe: Retrieval-native (prematched) VC training on Applio

End-to-end pipeline: fresh dataset -> preprocess -> features -> prematch ->
train (with spk + content losses, optional attention) -> evaluate.

Key idea: **prematching needs NO change to Applio's training code.** It replaces
each utterance's content features with features RETRIEVED from the same speaker's
OTHER utterances (kNN-VC style). Training then reads the prematched filelist as
normal. So "pass target features instead of source features" = a preprocessing
step, not a training-loop edit.

---

## 0. Which code to use (read this first)

Your working modifications (spk loss, content loss, null-key attention, the
`AUX` / `GTA` flags) are already on the server clone. Build the shareable repo
FROM that working clone -- do not reconstruct from scratch. See section 8 for
how to clean it and push to GitHub.

The ONLY new file needed for prematching is `prematch_features.py`
(in this repo). Everything else is your existing tested code.

---

## 1. Dataset layout

Multi-speaker: one integer-named folder per speaker under a dataset root:
```
dataset/
  0/  utt1.wav utt2.wav ...      # speaker 0, several utterances (needed for prematch)
  1/  ...
  ...
```
Each speaker needs SEVERAL utterances -- prematching retrieves from a speaker's
*other* utterances, so single-utterance speakers can't be prematched.

## 2. Preprocess (Applio)
```bash
python core.py preprocess \
    --model_name <exp> \
    --dataset_path dataset \
    --sample_rate 48000 \
    --cpu_cores 8
```
(check `python core.py preprocess --help` for exact arg names in your version)

## 3. Extract features (ContentVec + F0)
```bash
python core.py extract \
    --model_name <exp> \
    --f0_method rmvpe \
    --embedder_model contentvec \
    --sample_rate 48000
```
Produces the per-utterance ContentVec `.npy` features + F0, and writes
`logs/<exp>/filelist.txt` (format: `wav|feat|f0|f0nsf|sid`).

## 4. Build the FAISS index (the memory bank, for inference retrieval)
```bash
python core.py index --model_name <exp>
```

## 5. Precompute speaker targets (for the contrastive speaker loss)
```bash
export HF_HUB_DISABLE_XET=1
python precompute_spk_targets.py logs/<exp>          # -> logs/<exp>/spk_targets.pt
python3 -c "import torch;print(torch.load('logs/<exp>/spk_targets.pt').shape)"
```
Row count must be > your max speaker id.

## 6. PREMATCH the features (the retrieval-native step)
```bash
python prematch_features.py --filelist logs/<exp>/filelist.txt --k 4
```
Writes:
- `logs/<exp>/feature_prematched/<sid>_<idx>.npy`  (retrieved features)
- `logs/<exp>/filelist_prematched.txt`             (same rows, feat path -> prematched)

## 7. Train

Flags (env vars, all default OFF so normal training is unchanged):
- `AUX=1`   speaker + content losses ON
- `GTA=1`   vocoder-only finetune on predicted latents (needs converged pretrainG)
- `V2=1`    retrieval attention active (if built into synthesizers.py)

Three training conditions for the paper's comparison (one variable each):

```bash
# A) BASELINE: normal features, losses on
AUX=1 bash train.sh                       # filelist.txt

# B) PREMATCHED: retrieved features, losses on
#    -> point the run at filelist_prematched.txt (copy it in as filelist.txt in a
#       fresh exp folder, or set config.data.training_files to it)
AUX=1 bash train.sh                       # filelist_prematched.txt

# C) PREMATCHED + attention (the novelty variant), if V2 wired:
AUX=1 V2=1 bash train.sh                  # filelist_prematched.txt
```
Keep everything else identical (data, epochs, seed) so the only change is
normal-vs-prematched features. Record each run in a manifest.

## 8. Evaluate (vc-analysis toolkit)
```bash
# build eval set, chunk, then joint analysis of the trained models:
python analysis_vc.py \
    --source eval/source_wavs --ref eval/ref_wavs \
    --system baseline=eval/infer_baseline \
    --system prematch=eval/infer_prematch \
    --lang <code> --whisper large-v3 --out results_prematch
```
The Wilcoxon block tells you if prematch beats baseline on SECS / CER.

---

## 9. Make the clean GitHub repo (from your WORKING server clone)

On the server, in your working Applio clone:
```bash
cd /hdd4/Samara/<working_applio_clone>

# clean out data/checkpoints/caches so only code is pushed
cat > .gitignore <<'EOF'
logs/
*.pth
*.index
*.npy
*.wav
__pycache__/
*.pyc
EOF

# add the prematch tool alongside
cp /path/to/vc-analysis/prematch_features.py .
cp /path/to/vc-analysis/RECIPE_prematch_training.md README_prematch.md

git init
git add -A
git commit -m "Applio + spk/content losses, null-key attention, prematched training"
gh repo create applio-prematch-vc --public --source . --push
```
(or create the empty repo on github.com and `git remote add origin ... && git push`)

This pushes your TESTED code + the prematch tool + this recipe -- a clean,
reproducible research repo, without reconstructing anything.
