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
Business-impact analysis and Model Registry registration are next.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
