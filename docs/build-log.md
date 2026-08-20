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

## 2026-08-18 — Colab notebook scoped down; SeaWulf verified live

Three Colab T4 timing tests came back consistent regardless of batch size
or gradient checkpointing setting: 1.152, 1.068, 1.119 samples/sec --
a real ~1.1 samples/sec ceiling, not noise. Full dataset x 2 epochs
would need ~11 hours, too long for one Colab session. Updated
`notebooks/colab_train_qlora.ipynb` with a stratified 370/class
(~10,000 example) subsampling cell and cut the full-run cell to 1 epoch
over that subset (~2.5 hours), same methodology as the earlier local
subset script.

In parallel, checked whether SeaWulf (Stony Brook's HPC cluster) could
beat that instead of just accepting the Colab scope cut, using WinSCP's
built-in terminal over the existing SSH session rather than assuming
anything from public docs:

- The public SeaWulf docs the `seawulf_*.sbatch` scripts were originally
  written against turned out to be wrong on two counts: no `a100`
  partition exists on this cluster at all, and `module load anaconda3`
  isn't a real module name (it was silently failing via `|| true`,
  which would have left the venv built against the system's Python
  3.6.8 -- far too old for transformers/trl/peft/bitsandbytes).
- Checked real partitions via `sinfo`: `gpu`/`gpu-long`/`gpu-large`
  (nodes `sn-nvda[3-8]`) and `v100` (nodes `sn-nvda[1-2]`) are genuinely
  separate node pools, not the same hardware under different queue
  policies -- confirmed by requesting `nvidia-smi -L` on each via `srun`
  rather than guessing from partition/node names. Result: the `gpu`
  partition is **Tesla K80** (2014-era, no tensor cores -- likely
  *slower* than Colab's T4, not a real upgrade), while `v100` is
  **Tesla V100-PCIE-32GB** (real tensor cores, 32GB VRAM, 24hr limit).
- Fixed both sbatch scripts: `-p a100` -> `-p v100`, `--gpus=1` ->
  `--gres=gpu:1` (matches the exact flag verified working via `srun`),
  `module load anaconda3` -> `module load anaconda/3-new` (real module,
  confirmed via `module avail`, gives Python 3.11.15).
- Repo cloned onto SeaWulf at `/gpfs/home/mallabaksh/support-ticket-triage-llm-finetuning`.

Next: submit `scripts/seawulf_timing_check.sbatch` on the `v100`
partition, inspect real step times, and decide between the Colab subset
plan and a SeaWulf full-dataset run based on actual numbers -- same
discipline as every other compute decision so far.

## 2026-08-18 — SeaWulf abandoned: environment too old, not worth the fight

Submitted the timing check repeatedly and fixed real issues one at a
time, each confirmed live rather than guessed:

1. `pip install -r requirements.txt` tried to build numpy from source
   and failed -- node's system GCC is 4.8.5 (RHEL7-era), numpy needs
   >=9.3. Root cause: `--system-site-packages` (meant to reuse
   anaconda/3-new's precompiled numpy) doesn't help here because pip's
   build isolation for packages like scikit-learn/scipy ignores it and
   fetches its own numpy anyway. Fix: `requirements-seawulf.txt`
   containing only what `train_qlora.py` actually imports (pandas,
   datasets, transformers, accelerate, tqdm, peft, bitsandbytes, trl),
   dropping scikit-learn/scipy/matplotlib/seaborn entirely since
   training doesn't need them.
2. `bitsandbytes>=0.43` had no compatible wheel for this node (pip's own
   error listed available versions, capped at 0.42.0) -- relaxed the pin;
   0.42.0 still has full 4-bit NF4 support.
3. torch imported but crashed with a `libcusparse.so.12` /
   `__nvJitLinkAddData_12_1` symbol mismatch. Diagnosed by forcing pip to
   list every version it considered installable for
   `nvidia-nvjitlink-cu12` and `nvidia-cusparse-cu12` (both go up to
   CUDA 12.9 -- ruled out a version-availability problem). Real cause:
   `module load cuda120/toolkit/12.0` put the system's own older
   `libnvJitLink.so.12` ahead on `LD_LIBRARY_PATH`, which got loaded
   instead of the newer one torch's pip wheel bundles, regardless of
   what pip installed. Fix: stopped loading the system CUDA module
   entirely -- not needed since torch's wheel is self-contained.
   Confirmed fixed: GPU check then printed `CUDA: True` /
   `Tesla V100-PCIE-32GB` correctly.
4. `mlflow` (still in the requirements list at that point) pulls in
   scipy as a dependency, hitting the identical GCC-too-old build
   failure and aborting the whole install again. Dropped it -- both
   sbatch scripts run `--report-to none` and `train_qlora.py` never
   imports mlflow directly, so it wasn't needed.
5. Even after removing mlflow, the same scipy build error recurred,
   meaning something else in the requirements list also pulls it in
   transitively -- not yet isolated which package.

At that point, decided to stop rather than keep debugging this specific
node's ancient toolchain package-by-package. The V100 hardware itself
was confirmed real and fast (32GB VRAM, proper tensor cores), and every
fix so far was a genuine, verified root cause rather than a guess -- but
the cumulative time cost of fighting a RHEL7-era system environment
outweighed the speed benefit over Colab for a portfolio project.
`scripts/seawulf_timing_check.sbatch` and `scripts/seawulf_full_train.sbatch`
are left in the repo, fixed as far as they got, as a record of the
investigation and in case it's worth revisiting later.

Reverted to the Colab plan from the previous entry: stratified
10,000-example subset (370/class), 1 epoch, ~2.5 hours, already built in
`notebooks/colab_train_qlora.ipynb`.

## 2026-08-18 — QLoRA training completed on Colab; fine-tuned eval run locally

Ran the Colab plan: 313 steps, 1 epoch over the 9,990-example stratified
subset, took 2h22m (in line with the ~2.5hr estimate from measured
throughput). Training loss dropped smoothly from 0.966 (step 20) to
0.114 (step 300), no instability. Adapter downloaded and placed at
`models/qlora-adapter/`.

Colab's free-tier GPU quota was exhausted right after the long run, so
the fine-tuned evaluation was run locally instead
(`src/evaluation/finetuned_eval.py`, new) rather than waiting for quota
to reset -- eval is pure inference (270 short generations), nowhere near
the sustained load that caused local *training* to throttle earlier, so
local GPU handled it fine with no thermal issues (finished in ~3.5
minutes).

**Result**, same 270-example sample and prompt/parsing logic as
`baseline_eval.py` for a direct comparison:

| Metric | Baseline (zero-shot) | Fine-tuned |
|---|---|---|
| Accuracy | 55.6% | 100% |
| Macro F1 | 0.483 | 1.000 |
| Weighted F1 | 0.483 | 1.000 |
| Unparseable rate | 3.3% | 0% |

Perfect accuracy is a real result, not a bug -- verified by hand-checking
`results/finetuned/predictions.csv`: all five of the baseline's complete
failure categories (`edit_account`, `get_refund`,
`set_up_shipping_address`, `switch_account`, `track_refund`) predict
correctly now, and predictions span all 27 labels rather than
collapsing. Worth stating honestly, though: the Bitext dataset is
templated (placeholder slots, heavily repeated phrasing per intent), so
near-perfect accuracy on this specific held-out set reflects how
learnable this dataset's structure is under fine-tuning, not a claim
that the model would hit 100% on messier, non-templated real-world
tickets.

`README.md` and `docs/README.md` updated to match actual current state
(both had drifted -- still described the old CPU-only-fallback,
training-not-yet-run state from well before this whole compute saga).

Next: business-impact writeup, then register the adapter to the Azure
ML Model Registry.

## 2026-08-18 — Real data leakage found and fixed: dataset needed dedup before splitting

Before moving to the business-impact writeup, ran an interactive
qualitative test (`src/inference/classify_live.py`, new) -- typing
genuinely novel, non-templated messages at the fine-tuned model rather
than trusting the templated test set alone. Results were mixed in an
informative way: correct handling of negation and mid-sentence intent
pivots (e.g. "I don't want a refund, I just want to know where my
package is" -> `track_order`), but three clearly out-of-scope inputs
("what's the weather today", "write me a python function...") all got
confidently forced into a real (wrong) label instead of any "doesn't
apply" signal -- a genuine deployment-relevant limitation, not a dataset
artifact.

Separately, pushed back on the 100% accuracy / 1.000 macro F1 result
itself as implausible on its face -- correctly so. Investigated rather
than defended it:

- Checked for exact-text overlap between train and test: **405 of 4,031
  test rows (10%) had `instruction` text byte-identical to a training
  row.** Root cause: the dataset's `{{Placeholder}}` tokens are literal,
  unfilled strings, not substituted values, so many rows across the full
  26,872-row dataset are exact duplicates of each other (2,237 of them,
  confirmed always mapping to the same intent). `train_test_split` only
  guarantees disjoint *rows*, not disjoint *text* -- duplicate content
  leaks across the split regardless of a correct split implementation.
- Quantified impact on the actual 270-example eval sample specifically:
  **31/270 rows (11.5%) had exact-duplicate text in the training set.**
  This is real leakage and a real methodological flaw, but it doesn't
  fully explain the 100% result -- since overall accuracy was exactly
  270/270, the other 239 non-leaked rows were *also* all predicted
  correctly, meaning genuine fine-tuning-driven generalization accounts
  for the large majority of the result, not memorization of exact text.

Fixed at the source rather than just filtering the eval sample:
`src/data_prep/load_bitext.py` now deduplicates by `instruction` text
(keep first occurrence) before the stratified split. Confirmed zero
inconsistent labels among duplicates (safe to dedupe naively) and
healthy class balance afterward (24,635 unique rows, smallest class 493
examples). Regenerated the local split: 20,939 train / 3,696 test, zero
exact-text overlap confirmed. Applied the identical dedup step to
`notebooks/colab_train_qlora.ipynb`'s data-load cell so Colab's
independently-reloaded split matches exactly.

This requires a real redo, not just a note: baseline eval, training, and
fine-tuned eval were all computed on the old (leaky) split's specific
rows, so none of the existing numbers in `results/baseline/` or
`results/finetuned/` are valid against the new split's actual held-out
set. Sequence: (1) re-run `baseline_eval.py` on the new split, (2)
retrain on Colab with the new clean subset, (3) re-run
`finetuned_eval.py` locally on the new 270-example sample.

Step 1 also turned up a second, unrelated bug while re-running:
`baseline_eval.py` never actually put the model on GPU (no `device_map`
or `.to()` in its load call), so it silently ran on CPU the whole time
despite the local `.venv` having working CUDA -- explains why it always
took ~1.5hr instead of the few minutes `finetuned_eval.py` needed for
the same 270 examples. Fixed (`device_map="auto"`, moved inputs to
`model.device`). First GPU re-run crashed with a segfault (exit 139);
root cause was a leftover `classify_live.py` interactive session from
earlier still holding the model in GPU memory, fighting the new process
for CUDA context on the 6GB card. Killed the stray process, GPU memory
confirmed clear, re-ran cleanly in 2m16s.

**New baseline (deduplicated split):**

| Metric | Old (leaky) | New (clean) |
|---|---|---|
| Accuracy | 55.6% | 58.9% |
| Macro F1 | 0.483 | 0.538 |
| Weighted F1 | 0.483 | 0.538 |
| Unparseable rate | 3.3% | 4.1% |

Small shift, as expected from different specific rows landing in the new
270-example sample -- nothing alarming. Next: retrain on Colab with the
deduplicated subset, then re-run the fine-tuned eval for the real,
leak-free before/after comparison.

## 2026-08-20 — Retrained on the deduplicated split; real leak-free numbers

Reconnected to Colab (GPU quota had reset) and re-ran the full notebook
top to bottom on the deduplicated dataset: same dedup counts as the
local run (26,872 -> 24,635 rows, 20,939 train / 3,696 test, confirming
the split is reproducible across environments), 9,990-example subset,
313 steps, ~2h20m, loss 0.968 -> 0.117 -- same clean convergence pattern
as the first run. Ran the in-notebook eval cell against the new adapter
on the new 270-example sample while still connected (no separate local
step needed this time), downloaded both the adapter and eval results,
moved into `models/qlora-adapter/` and `results/finetuned/`.

**Real result:**

| Metric | Baseline (clean) | Fine-tuned (clean) |
|---|---|---|
| Accuracy | 58.9% | 99.6% (269/270) |
| Macro F1 | 0.538 | 0.996 |
| Weighted F1 | 0.538 | 0.996 |
| Unparseable rate | 4.1% | 0% |

269/270, not a suspicious 270/270 -- this is the reassuring outcome the
whole leakage investigation was aiming for. The one miss: *"error
opening user profile"* labeled `registration_problems`, predicted
`edit_account` -- a genuinely ambiguous message, not a model failure.
Checked by hand rather than trusted from the aggregate metric alone,
same discipline as the earlier (leaky) 100% result.

`README.md` and `docs/README.md` updated with the final numbers,
replacing the placeholder/pending text from the leakage-investigation
entries. This closes out Project 2a's core training and evaluation work.

Next: business-impact writeup, then register the adapter to the Azure
ML Model Registry.
