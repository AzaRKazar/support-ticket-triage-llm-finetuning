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

## 2026-08-17 — GPU quota still blocked; pivoted to local GPU training

Re-checked Azure GPU quota (all 31 eligible regions again): still 0
everywhere. Investigated why rather than just re-confirming the fact --
subscription's `quotaId` is `PayAsYouGo_2014-09-01` (checked via
`az rest` against the subscription resource directly), which rules out
the free-trial dead end. Most likely explanation: the earlier "support
confirmed the VM series" response was general technical support
confirming the SKU choice, not the separate, slower quota-approval queue
that actually grants it -- confirmed this distinction is real by
attempting the request through both the Portal's self-service Quotas
blade and directly through the `az quota` API; both failed identically
with `QuotaNotAvailableForResource`, meaning this specific GPU family
requires a formal support ticket regardless of path. Filed one correctly
this time (issue type "Service and subscription limits (quotas)", not
a generic request) and left it pending -- Azure's turnaround, not
something to keep re-checking.

Rather than continue waiting, checked whether local training was
actually viable instead of assuming CPU-only: this machine turned out to
have an NVIDIA GTX 1660 Ti (Windows' WMI reported 4GB VRAM, which is a
known 32-bit overflow bug in that field for cards with more --
`nvidia-smi` confirmed the real number, 6GB). Swapped the CPU-only torch
build for a CUDA build (`cu121` wheels) and ran a smoke test
(`src/training/smoke_test_gpu.py`): loaded Qwen2.5-1.5B-Instruct in
4-bit, attached LoRA adapters, ran one real forward+backward pass. Peak
VRAM: 2.09GB out of 6.44GB available -- comfortable headroom, not a
tight fit. Local QLoRA training is genuinely viable, not just
theoretically possible.

This doesn't abandon the Azure architecture -- it's the same fallback
pattern the original brief already sanctioned for Colab (train
elsewhere, register the resulting adapter to Azure ML afterward), just
with a better fallback than Colab: no time limits, no session handling,
and the training run's MLflow logging can still point at the Azure ML
workspace's tracking URI remotely, so it still shows up in the
workspace's experiment tracking even though the GPU doing the work isn't
Azure's.

Next: run the real training job for Project 2a on the full 22,841
example training set.

## 2026-08-17 — Local GPU throttling confirmed severe; pivoted to Colab

Fixed a real bug found in the timing test: `trl` 1.10.0 renamed
`SFTConfig`'s `max_seq_length` to `max_length` -- the script was written
against an earlier API. Checked the rest of the config's parameter names
against the installed version's actual signature before re-running,
rather than fix errors one at a time.

First real timing test (default config, full 22,841-example data): 20
steps took 12.6 minutes, extrapolating to **~30 hours** for a full 2-epoch
run -- not viable. Investigated why rather than just picking smaller
numbers: found the data's actual token length (mean 145, max 157) was
a fraction of the configured `max_seq_length=512`, so raised batch size
and cut sequence length to match the real data, backed by a token-length
check rather than a guess.

That fix made things *worse* on the immediate retest (67-85 sec/step vs.
the original 19-42 sec/step) -- checked `nvidia-smi` live rather than
assume the config change was bad, and found the real cause: GPU at 76C,
clocked at 1080MHz against a 2100MHz max, drawing only 27W. Classic
thermal/power throttling from two back-to-back tests with no cooldown.

The laptop then powered off unexpectedly mid-test. Restarted, confirmed
no orphaned processes, and re-tested from a cooler baseline (68C) on a
smaller 1,620-example stratified subset (`src/data_prep/build_local_subset.py`,
60/class) specifically to get a clean reading. Same pattern recurred:
step time climbed steadily within the first 10 steps (39s -> 77s) even
starting cooler -- this GPU throttles within minutes under any sustained
load, not after a long warm-up. Steady-state rate settled around
~10 sec/sample both times, consistent enough to trust as the real number:
even the "safe" 1,620-example subset would need 4+ hours at that rate,
not the 15-20 minutes originally estimated from the optimistic early-step
numbers.

This is a genuine hardware limitation (6GB laptop GPU, WDDM driver
overhead, thermal design not built for sustained ML workloads), not a
config problem to keep tuning around. Three options were on the table:
a much smaller (~120-150 example) subset for one short session, burst
training with cooldown pauses using checkpointing, or moving training to
Google Colab. Chose Colab first, since the brief already sanctioned it as
a fallback and Colab's T4 has proper server cooling with no local
thermal ceiling -- built `notebooks/colab_train_qlora.ipynb`, which
rebuilds the identical train/test split (same dataset, same
`random_state=42`) rather than requiring a manual data upload, and
includes its own bounded timing check before committing to a full run,
same discipline as the local attempts. If Colab also can't sustain the
full dataset, burst+cooldown local training is the fallback.

Next: run the notebook on Colab, get a real (not assumed) timing
measurement, then decide on full-dataset vs. reduced scope based on that.
