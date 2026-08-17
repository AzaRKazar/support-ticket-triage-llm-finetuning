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

In progress. Done so far: Azure ML workspace + compute cluster provisioned,
dataset loaded/explored/split, zero-shot baseline evaluation running,
QLoRA training script drafted (not yet run — waiting on GPU quota).
See [`docs/build-log.md`](docs/build-log.md) for the running engineering
log, including a couple of real mistakes made and caught along the way.
Metrics, confusion matrix, and business-impact analysis will replace this
section once training and evaluation are complete.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
