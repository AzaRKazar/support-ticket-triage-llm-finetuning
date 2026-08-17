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

**Why CPU compute, not GPU:** the brief's plan called for an
NCasT4_v3-family GPU cluster for QLoRA training. Actual quota (checked via
`az vm list-usage` across all 31 regions offering that SKU) came back 0
everywhere — the support ticket confirming "VM series correct" was not
confirmation of an approved grant. Rather than block on that ticket, the
pipeline is being built end-to-end on CPU compute now (workspace,
tracking, registry, endpoint), with the GPU-dependent training step ready
to run the moment quota lands. See `build-log.md` for the full quota-check
detail.

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

**Status:** script written (`src/training/train_qlora.py`), **not yet
executed anywhere**. 4-bit quantization requires CUDA, so this can't be
smoke-tested on the local CPU machine — it's built to standard,
well-established QLoRA patterns, but unverified until real GPU compute is
available.

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

**Status:** running as of this writing. Results (`results/baseline/`)
will be added here once complete, followed by the same evaluation run
against the fine-tuned model for direct comparison.

## Key decisions and why

| Decision | Why |
|---|---|
| CPU-only fallback instead of waiting on GPU quota | Ticket confirmation ≠ granted quota; verified 0 across all eligible regions. Fallback path keeps the rest of the pipeline (workspace, tracking, registry, endpoint) moving. |
| eastus region | Brief's suggested default; not GPU-quota-driven since no region has quota yet. Revisit if/when quota lands elsewhere. |
| Stratified 270-example eval sample | CPU-only inference is too slow for full-test-set iteration (~20 sec/example). Documented explicitly rather than presented as the full set. |
| Same base model for baseline and fine-tune | Required for the before/after comparison to isolate the effect of fine-tuning rather than model choice. |
| Compute Cluster, not Compute Instance | Instances are always-on single VMs with no autoscale-to-zero; caught and corrected after an instance was accidentally left running. See `build-log.md`. |

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/data_prep/load_bitext.py       # loads data, writes train/test split
python src/evaluation/baseline_eval.py    # zero-shot baseline metrics
# src/training/train_qlora.py requires GPU compute (Azure ML job) -- see docs/build-log.md
```
