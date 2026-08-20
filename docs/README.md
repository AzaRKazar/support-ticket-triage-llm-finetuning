# Project Documentation

Consolidated explainer covering the data, infrastructure, training
approach, and evaluation approach for this project, and the reasoning
behind the non-obvious decisions. For a chronological, as-it-happened
account (including mistakes made and fixed), see
[`build-log.md`](build-log.md). This file gets updated as the project
progresses; it does not yet reflect a finished project.

## The problem

Support teams route tickets manually or with brittle keyword rules. This
project fine-tunes a small instruction-tuned LLM to classify incoming
tickets by intent (billing, refunds, shipping, account issues, etc.),
measured against a zero-shot baseline on the same base model, and served
through a real Azure ML pipeline rather than a one-off notebook.

## Data

**Source:** [Bitext Customer Support LLM Chatbot Training
Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(public, Hugging Face).

**Shape:** 26,872 examples across 27 intent categories (`check_invoice`,
`complaint`, `cancel_order`, `track_refund`, etc.), each with the
customer's message (`instruction`), a broader `category` grouping, the
specific `intent` label, and a reference `response`. Only `instruction`
and `intent` are used for classification training/eval — `category` and
`response` aren't currently used but are kept in the raw data in case
they're useful for later error analysis.

**Class balance:** notably even — every category has 950-1,000 examples.
No class-imbalance handling (oversampling, class weighting) was needed as
a result.

**Split:** 85/15 stratified by intent → 22,841 train / 4,031 test, fixed
`random_state=42` for reproducibility. Saved locally as parquet
(gitignored — regenerate anytime via `src/data_prep/load_bitext.py`, it
re-downloads from Hugging Face and re-splits deterministically).

## Infrastructure (Azure ML)

| Resource | Value |
|---|---|
| Resource group | `rg-finetuning-projects` (eastus) |
| Workspace | `mlw-finetuning` |
| Compute | `cpu-cluster` — Standard_DS3_v2, min 0 / max 1 nodes |
| Budget alert | $35/month, 80% actual-spend threshold |
| MLflow tracking | Native to the workspace, no separate setup |

**Why the actual training didn't run on Azure ML compute:** the brief's
plan called for an NCasT4_v3-family GPU cluster for QLoRA training.
Actual quota (checked via `az vm list-usage` across all 31 regions
offering that SKU) came back 0 everywhere — a formal support ticket was
filed and is still pending Azure's turnaround. Rather than block on that,
training moved through a documented sequence of fallbacks instead: local
GPU (viable in principle, but hit reproducible thermal throttling under
sustained load — see `build-log.md`), then Google Colab (where training
actually ran to completion), with a SeaWulf HPC detour explored and
abandoned along the way (real V100 GPUs, but an environment too old to
be worth fighting for a portfolio project). The rest of the pipeline —
workspace, tracking, registry, endpoint — is still built on Azure ML;
only the compute-heavy training step ran elsewhere, with the resulting
adapter brought back to Azure ML's Model Registry afterward.

## Training approach

**Method:** QLoRA — 4-bit NF4 quantization (`bitsandbytes`) + LoRA
adapters (`peft`) on the attention and MLP projection layers
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
`down_proj`), trained via `trl`'s `SFTTrainer`.

**Model:** Qwen2.5-1.5B-Instruct (also evaluated zero-shot as the
baseline, for a fair before/after comparison against the same base
model).

**Data format:** each training example becomes a 3-turn chat: a system
prompt listing all 27 valid intent labels, the customer's message as the
user turn, and the correct intent label as the assistant turn — the model
is trained to answer with just the label.

**Tracking:** `report_to="mlflow"` in the training config. Azure ML sets
`MLFLOW_TRACKING_URI` automatically for jobs run in the workspace, so
params/metrics/the adapter log to the workspace's MLflow tracking with no
extra wiring required.

**Status:** complete. Trained on Google Colab's free-tier T4 GPU (`notebooks/colab_train_qlora.ipynb`),
on a stratified 370-examples-per-class subset (9,990 total) for 1 epoch —
a deliberate scope reduction from the original full-dataset/2-epoch
plan, based on measured Colab throughput (~1.1 samples/sec, confirmed
consistent across three separate timing tests) that made the full run's
~11-hour estimate impractical for one Colab session. Trained twice: an
initial run, then a full retrain (same notebook, same hyperparameters,
~2h20m, 313 steps, loss 0.968 → 0.117) after a real data-leakage bug was
found and fixed at the source (dataset deduplicated by `instruction`
text before splitting — see `build-log.md`, 2026-08-18). Adapter
downloaded and stored locally at `models/qlora-adapter/` (gitignored —
see `build-log.md` for the full compute-fallback story: local GPU
throttling, the Colab pivot, and a SeaWulf detour that was explored and
abandoned).

## Evaluation approach

**Baseline (zero-shot):** the un-fine-tuned base model is prompted with
the same system-prompt format described above (list of valid intents +
customer message) and asked to output one label. Its raw text output is
matched against the valid label set (exact match, then substring match,
then marked unparseable) to get a predicted intent.

**Sampling:** evaluated on a **stratified sample of 270 examples** (10
per class) from the 3,696-example test set (post-dedup), not the full
set — full-set inference at this rate would still take a while, and 270
is enough to see real signal. Runs on local GPU (`device_map="auto"`,
fixed 2026-08-18 — it silently ran on CPU for a while before that,
~1.5hr instead of ~2min). Same 270-example sample reused for the
fine-tuned model's evaluation so the before/after comparison stays
apples-to-apples.

**Metrics:** accuracy, macro F1, weighted F1, per-class precision/recall/F1
(`sklearn.metrics.classification_report`), and a full confusion matrix.
Unparseable-output rate is also tracked, since a base model with no
fine-tuning may not reliably follow the "respond with exactly one label"
instruction.

**Status:** complete. Full results in `results/baseline/` (metrics.json,
predictions.csv, confusion_matrix.png).

| Metric | Value |
|---|---|
| Accuracy | 58.9% |
| Macro F1 | 0.538 |
| Weighted F1 | 0.538 |
| Unparseable output rate | 4.1% |

(Re-run 2026-08-18 on a deduplicated split -- an earlier version had
405/4,031 test rows with exact-duplicate `instruction` text also in
train, since the dataset's `{{Placeholder}}` tokens are literal unfilled
strings rather than substituted values. See `build-log.md` for the full
investigation.)

Strongest category: `payment_issue` (F1 1.000). Complete failures
(F1 = 0.0): `edit_account`, `get_invoice`, `get_refund`,
`set_up_shipping_address` — the base model follows the output-format
instruction fine but can't separate this dataset's fine-grained category
boundaries, exactly the gap fine-tuning is meant to close.

**Fine-tuned model:** evaluated on the identical 270-example sample
(same rows, same prompt format, same parsing logic) with the trained
QLoRA adapter applied — direct apples-to-apples comparison against the
baseline above. Trained and evaluated on the deduplicated split (see
`build-log.md`, 2026-08-18 entries, for the full leakage investigation
that led here). Full results in `results/finetuned/`.

**Status:** complete.

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Accuracy | 58.9% | 99.6% (269/270) |
| Macro F1 | 0.538 | 0.996 |
| Weighted F1 | 0.538 | 0.996 |
| Unparseable output rate | 4.1% | 0% |

99.6%, not a suspicious 100% — genuinely reassuring, since the earlier
(leaky-split) run's perfect score was exactly the kind of result that
should raise an eyebrow, and turned out to have real (if partial)
leakage behind it. The one miss here is a legitimate, understandable
boundary case: *"error opening user profile"* — true label
`registration_problems`, predicted `edit_account`. Both are reasonable
readings of an ambiguous message, not a model malfunction.

**Live qualitative testing (`src/inference/classify_live.py`):** on the
(now-superseded, leaky-split) adapter, handled negation and mid-message
intent pivots correctly, but had no ability to reject genuinely
out-of-scope input — three unrelated messages ("what's the weather
today", "write me a python function...") all got confidently forced
into a real, wrong label rather than any "doesn't apply" signal. Not
yet re-tested on the clean-split adapter, but worth keeping as a stated
limitation in the business-impact writeup regardless — a deployed
version needs a confidence threshold or explicit fallback, not blind
trust in the 27-label output.

## Key decisions and why

| Decision | Why |
|---|---|
| CPU-only fallback instead of waiting on GPU quota | Ticket confirmation ≠ granted quota; verified 0 across all eligible regions. Fallback path keeps the rest of the pipeline (workspace, tracking, registry, endpoint) moving. |
| eastus region | Brief's suggested default; not GPU-quota-driven since no region has quota yet. Revisit if/when quota lands elsewhere. |
| Stratified 270-example eval sample | CPU-only inference is too slow for full-test-set iteration (~20 sec/example). Documented explicitly rather than presented as the full set. Same sample reused for the fine-tuned model. |
| Same base model for baseline and fine-tune | Required for the before/after comparison to isolate the effect of fine-tuning rather than model choice. |
| Compute Cluster, not Compute Instance | Instances are always-on single VMs with no autoscale-to-zero; caught and corrected after an instance was accidentally left running. See `build-log.md`. |
| Training moved off Azure ML compute (local GPU, then Colab) | GPU quota never landed; local GPU throttled under sustained load (confirmed 3x via `nvidia-smi`); Colab (the brief's own sanctioned fallback) had proper cooling and gave consistent, usable throughput. |
| 10,000-example subset, 1 epoch (not full 22,841 x 2 epochs) | Measured Colab throughput (~1.1 samples/sec, consistent across 3 tests) made the full run ~11 hours — too long for one Colab session. Loss curve during testing converged fast, suggesting the full dataset wasn't needed to show a real improvement. |
| SeaWulf explored, then abandoned | Real V100 GPUs confirmed live, but the node's RHEL7-era toolchain (GCC 4.8.5) made getting a working ML environment impractical relative to the time saved over Colab. |
| Fine-tuned eval run locally, not Colab | Colab's free-tier GPU quota was exhausted after the long training run. Eval is pure inference (a few hundred short generations) — nowhere near the sustained load that caused local training to throttle, so local GPU handles it fine. |

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data_prep/load_bitext.py       # loads data, writes train/test split
python src/evaluation/baseline_eval.py    # zero-shot baseline metrics
# src/training/train_qlora.py requires GPU compute (Azure ML job) -- see docs/build-log.md
```
