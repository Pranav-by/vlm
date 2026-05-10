# Real Pipeline — Lightning AI with T4 GPU

## Quick Start

### 1. Go to [Lightning AI](https://lightning.ai) and create a free account

### 2. Create a new Studio with **T4 GPU**

### 3. Upload this entire `real_pipeline/` folder to Lightning AI

### 4. Open a terminal and run:

```bash
cd real_pipeline
pip install -r requirements_lightning.txt
python run_on_lightning.py
```

### 5. When complete, download the `real_data/` folder to your local machine:
```bash
# On your local machine:
# Use Lightning AI's download feature or SCP
# Place it at ~/vlm_mia/real_data/
```

### 6. Run inference attacks locally:
```bash
conda activate vlm_mia
bash real_pipeline/07_run_attacks_local.sh
```

## What This Does

1. Downloads `llava_instruct_150k.json` from HuggingFace (~30MB, full 158K entries)
2. Selectively downloads ~500 COCO images (~50MB, NOT the full 18GB)
3. Splits data: 80% member (400) / 20% non-member (100) + 4-way shadow split
4. LoRA fine-tunes TWO LLaVA-7B models (shadow + target) on their respective member data
5. Generates conversations at multiple temperatures
6. Computes Rouge-2 + MPNet similarity scores
7. Saves everything to `real_data/` for local attack execution

## Time Estimate: ~3 hours on T4 (free tier)

| Step | Time |
|------|------|
| Setup + download | ~10 min |
| Data split | ~1 min |
| LoRA fine-tune (2 models × 1 epoch) | ~30 min |
| Conversation generation | ~90 min |
| Similarity computation | ~30 min |
| **Total** | **~2.5-3 hours** |

> The script is **resumable** — saves progress every 25 samples. Re-run if session times out.

## Disk Usage: <1 GB

| Item | Size |
|------|------|
| LLaVA-7B 4-bit model | ~4 GB (cached, temporary) |
| COCO images subset (500) | ~50 MB |
| Dataset JSONs | ~30 MB |
| LoRA adapters (2×) | ~50 MB |
| Conversation outputs | ~50 MB |
| Similarity scores | ~10 MB |
| **Your data to download** | **~140 MB** |
