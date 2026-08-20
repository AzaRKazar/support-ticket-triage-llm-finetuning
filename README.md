# Support Ticket Triage — LLM Fine-Tuning

Fine-tuning a small open LLM to classify customer support tickets into intent
categories (billing, refunds, shipping, account issues, etc.), deployed as a
real Azure ML pipeline.

## Problem

Support teams route tickets manually or with brittle keyword rules. This
project fine-tunes a small instruction-tuned LLM to classify incoming tickets
by intent, measured against a zero-shot baseline on the same model.

## Dataset

[Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
(public, Hugging Face) — ~27 intent categories, thousands of labeled examples.

## Approach

- **Model:** Qwen2.5-1.5B-Instruct or Llama-3.2-3B-Instruct
- **Method:** QLoRA (4-bit quantization + LoRA adapters)
- **Platform:** Azure ML — workspace, managed compute, tracked training jobs,
  MLflow experiment tracking, model registry, managed online endpoint

## Architecture

```text
Azure ML Workspace -> Compute Cluster -> Training Job (QLoRA) -> MLflow tracking
                                                                       |
                                                                       v
                                                          Model Registry -> Managed Online Endpoint
```

## Status

In progress. QLoRA training and evaluation are complete:

| Metric | Zero-shot baseline | Fine-tuned |
|---|---|---|
| Accuracy | 58.9% | 99.6% (269/270) |
| Macro F1 | 0.538 | 0.996 |
| Weighted F1 | 0.538 | 0.996 |

(Same 270-example stratified sample for both, drawn from a deduplicated
train/test split — an earlier version of this split had real text
leakage between train and test, caught, investigated, and fixed at the
source rather than papered over; see `docs/build-log.md`, 2026-08-18.
The one fine-tuned miss is a genuinely ambiguous case, not a model
malfunction.) Training ran on Google Colab, not Azure ML compute — Azure
GPU quota never landed, local GPU throttled under sustained load, and a
SeaWulf HPC detour was explored and abandoned; the full story is in the
build log.

See [`docs/build-log.md`](docs/build-log.md) for the running engineering
log, including several real mistakes made and caught along the way, and
[`docs/README.md`](docs/README.md) for the consolidated reference.
Model Registry registration is next.

## Business impact

**The problem in practice:** before a support ticket can be resolved, it
has to reach the right queue — billing, shipping, account access, refunds,
and so on. Teams typically do this by hand (an agent reads and routes
every ticket before anyone works it) or with keyword rules (fast, but
brittle — breaks on paraphrasing, typos, or indirect phrasing). Both cost
either labor or accuracy.

**What the numbers mean:** the zero-shot base model routed correctly
58.9% of the time — not reliable enough to trust unsupervised; a human
would still need to check roughly 4 of every 10 routing decisions, which
mostly cancels out the time saved. After fine-tuning, that jumped to
99.6% (269/270) on the held-out sample. At that accuracy, the realistic
deployment pattern is **auto-route the model's prediction and skip
manual triage entirely for the large majority of tickets** — turning a
per-ticket human task into a per-ticket machine one, with agent time
freed up for actually resolving issues instead of sorting them.

**What this doesn't cover, stated plainly:**

- The 99.6% figure is measured on a 270-example sample from the same
  dataset the model trained on (albeit a genuinely held-out, deduplicated
  split — see `docs/build-log.md` for how seriously that was checked).
  It is not a measurement on this company's or any real production
  ticket stream, which will include phrasing, slang, and edge cases this
  dataset doesn't represent.
- Interactive testing (`src/inference/classify_live.py`) found the model
  has **no ability to recognize input that doesn't belong to any of the
  27 categories** — genuinely unrelated messages get confidently forced
  into a real (wrong) label rather than flagged. A production deployment
  needs a confidence threshold or an explicit "route to human" fallback
  layer on top of this model, not a bare `argmax` over its output.
- The one miss in the clean eval sample was a legitimately ambiguous
  message a human could plausibly also misroute — a reminder that even
  a near-perfect router won't hit 100% on real, messy input, and the
  remaining human-review budget should be sized for genuine ambiguity,
  not treated as a rounding error to eliminate.

**Bottom line:** fine-tuning took routing accuracy from "needs checking"
to "trustworthy for the common case," which is where the real labor
savings live — but shipping this safely means pairing it with an
out-of-scope/low-confidence fallback, not deploying the raw model as a
black box.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
