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
on a stratified 370-examples-per-class subset (9,990 total, ~44% of the
full 22,841-example training set) for 1 epoch — a deliberate scope
reduction from the original full-dataset/2-epoch plan, based on measured
Colab throughput (~1.1 samples/sec, confirmed consistent across three
separate timing tests) that made the full run's ~11-hour estimate
impractical for one Colab session. Training took 2h22m across 313 steps;
training loss dropped from 0.966 (step 20) to 0.114 (step 300) with
smooth, stable convergence. Adapter downloaded and stored locally at
`models/qlora-adapter/` (gitignored — see `build-log.md` for the full
compute-fallback story: local GPU throttling, the Colab pivot, and a
SeaWulf detour that was explored and abandoned).

## Evaluation approach

**Baseline (zero-shot):** the un-fine-tuned base model is prompted with
the same system-prompt format described above (list of valid intents +
customer message) and asked to output one label. Its raw text output is
matched against the valid label set (exact match, then substring match,
then marked unparseable) to get a predicted intent.

**Sampling:** evaluated on a **stratified sample of 270 examples** (10
per class) from the 4,031-example test set, not the full set. This
machine is CPU-only with no native bf16 acceleration, and full-set
inference was projected at many hours; the 270-example sample itself
takes roughly 1.5 hours. This is a deliberate, documented tradeoff, not a
hidden shortcut — the same 270-example sample will be reused for the
fine-tuned model's evaluation so the before/after comparison stays
apples-to-apples, even though it's a sample rather than the full test set.

**Metrics:** accuracy, macro F1, weighted F1, per-class precision/recall/F1
(`sklearn.metrics.classification_report`), and a full confusion matrix.
Unparseable-output rate is also tracked, since a base model with no
fine-tuning may not reliably follow the "respond with exactly one label"
instruction.

**Status:** complete. Full results in `results/baseline/` (metrics.json,
predictions.csv, confusion_matrix.png).

| Metric | Value |
|---|---|
| Accuracy | 55.6% |
| Macro F1 | 0.483 |
| Weighted F1 | 0.483 |
| Unparseable output rate | 3.3% |

Strongest categories (F1 ~0.95): `check_cancellation_fee`,
`payment_issue`, `recover_password`. Complete failures (F1 = 0.0):
`edit_account`, `get_refund`, `set_up_shipping_address`,
`switch_account`, `track_refund` — all five turned out to be systematic
collapses into a semantically adjacent category rather than random noise
(e.g. `set_up_shipping_address` was predicted as `change_shipping_address`
100% of the time; `track_refund` as `check_refund_policy` 100% of the
time). The base model follows the output-format instruction fine but
can't separate this dataset's fine-grained category boundaries — the
gap fine-tuning is meant to close.

**Fine-tuned model:** evaluated on the identical 270-example sample
(same rows, same prompt format, same parsing logic) with the trained
QLoRA adapter applied, so this is a direct apples-to-apples comparison
against the baseline above. Run locally on GPU (`src/evaluation/finetuned_eval.py`)
rather than Colab, since this is pure inference — a few hundred short
generations — nowhere near the sustained load that caused local
*training* to throttle. Full results in `results/finetuned/`.

**Status:** complete.

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Accuracy | 55.6% | 100% |
| Macro F1 | 0.483 | 1.000 |
| Weighted F1 | 0.483 | 1.000 |
| Unparseable output rate | 3.3% | 0% |

All five of the baseline's complete-failure categories (`edit_account`,
`get_refund`, `set_up_shipping_address`, `switch_account`,
`track_refund` — F1 0.0 zero-shot, each systematically collapsing into a
semantically adjacent category) predict perfectly after fine-tuning, and
predictions span the full 27-label space rather than collapsing —
verified by inspecting `results/finetuned/predictions.csv` directly
rather than trusting the aggregate metric alone.

**Caveat worth stating plainly:** the Bitext dataset is templated/
synthetic (placeholder slots like `{{Order Number}}`, heavily repeated
phrasing patterns per intent — e.g. "cancel purchase X" / "help me
canceling purchase X" for the same label). A fine-tuned model reaching
near-perfect accuracy on this specific 270-example held-out set is
genuine (confirmed above, not a parsing artifact), but it reflects how
learnable this dataset's structure is, not a guarantee of 100% accuracy
on messier, non-templated real-world support tickets. The honest
interpretation: fine-tuning clearly and dramatically closed the gap the
baseline showed, not that the model is flawless on all inputs.

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
