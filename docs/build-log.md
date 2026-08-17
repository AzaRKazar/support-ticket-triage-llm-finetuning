# Build Log

Running log of what was done and why, in the order it happened. The
polished case-study writeup (metrics, business impact) will live in the
main README once training and evaluation are complete — this file is the
working engineering journal behind it.

## 2026-08-17 — GPU quota check

Requested NCasT4_v3 (Standard NCASv3_T4 Family) quota increase; Azure
support confirmed the VM series choice was correct, but that only
validates the request, not the grant. Checked actual quota via
`az vm list-usage` across all 31 Azure regions that offer the
`Standard_NC4as_T4_v3` SKU: **every region showed a limit of 0.**
Decision: proceed with the CPU-only fallback path rather than block on
the support ticket resolving.

## 2026-08-17 — Azure ML workspace + compute

- Resource group `rg-finetuning-projects` and workspace `mlw-finetuning`
  created in `eastus` (the brief's suggested default; not tied to any
  approved GPU region since none exists yet). Workspace auto-provisioned
  a storage account, key vault, Log Analytics workspace, and Application
  Insights, and comes with a working MLflow tracking URI out of the box.
- Budget alert `finetuning-budget` set at $35/month, 80% actual-spend
  threshold, alerting the account email.
- First compute attempt accidentally created a **Compute Instance**
  (`cpucluster2`, Standard_E4ds_v4) instead of a **Compute Cluster** —
  instances are single always-on VMs with no min/max node autoscaling, so
  it was sitting there billing with no way to reach "0 min nodes." Caught
  via direct CLI verification (`az ml compute show`), stopped immediately
  (cost impact: a few minutes at ~$0.30-0.40/hr, negligible), then
  deleted. Correct resource created afterward: compute cluster
  `cpu-cluster`, Standard_DS3_v2, min instances 0, max 1 — verified
  directly via CLI rather than trusting the UI summary screen.

**Lesson:** Azure ML Studio's "Compute instances" and "Compute clusters"
are separate tabs that are easy to conflate; only clusters scale to zero.

## 2026-08-17 — Bitext dataset

Loaded via Hugging Face `datasets`: 26,872 examples, 27 intent categories,
notably well-balanced (950-1000 examples per class, no class-imbalance
handling needed). Split 85/15 stratified by intent → 22,841 train /
4,031 test, saved locally as parquet (gitignored, regenerable via
`src/data_prep/load_bitext.py`).

## 2026-08-17 — Zero-shot baseline evaluation

Evaluated the un-fine-tuned Qwen2.5-1.5B-Instruct via prompted
classification (system prompt lists all 27 valid intents, model must
respond with exactly one). Ran on a **stratified sample of 270**
(10 per class), not the full 4,031-example test set — this machine is
CPU-only (i7-10750H, no native bf16 acceleration) and full-set inference
would take many hours. The same 270-example sample will be reused for the
fine-tuned model's evaluation later so the before/after comparison is
apples-to-apples. This is a real methodological tradeoff, not a hidden
one — noting it here and in the final results writeup.

**Result:** took 1h21m. Accuracy 55.6%, macro F1 0.483, weighted F1
0.483, 3.3% unparseable outputs. Strongest categories (F1 ~0.95):
`check_cancellation_fee`, `payment_issue`, `recover_password`. Complete
failures (F1 = 0.0): `edit_account`, `get_refund`,
`set_up_shipping_address`, `switch_account`, `track_refund` — inspecting
predictions showed these aren't random errors but systematic collapses
into semantically adjacent categories (`set_up_shipping_address` →
100% predicted as `change_shipping_address`; `track_refund` → 100%
predicted as `check_refund_policy`; `switch_account` → 80% predicted as
`change_order`). The base model can follow the instruction format but
can't distinguish this dataset's fine-grained category boundaries —
exactly the gap fine-tuning is meant to close. Full metrics, per-class
report, predictions, and confusion matrix in `results/baseline/`.

## 2026-08-17 — QLoRA training script

Drafted `src/training/train_qlora.py` (4-bit NF4 quantization + LoRA via
peft/trl, `report_to="mlflow"` for automatic Azure ML tracking) while the
baseline eval ran in the background. **Not yet executed anywhere** —
bitsandbytes 4-bit quantization requires CUDA, unavailable until GPU
quota is actually granted. Written to standard, well-established QLoRA
patterns, but unverified until it runs on real GPU compute.
