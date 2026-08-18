#!/bin/bash
#SBATCH -J qlora-full-train
#SBATCH -p v100
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -o slurm-full-%j.out
#SBATCH -e slurm-full-%j.err

# Step 2 of 2: full training run. Only submit this AFTER
# seawulf_timing_check.sbatch has completed and its step times looked
# stable, not climbing. If they climbed like the local runs did, tell
# Claude the numbers before submitting this -- the plan may need to
# change (smaller data, more time, a different queue) rather than just
# running this blind.
#
# Verified live on SeaWulf 2026-08-18: no `a100` partition exists on this
# cluster. Using `v100` (Tesla V100-PCIE-32GB), whose real ceiling is a
# 24-hour limit in the same partition -- no separate "-long" queue needed.
# 8-hour wall time requested here as a starting budget; if the timing
# check's per-step rate implies the full run needs longer, raise -t (up
# to 24:00:00) before submitting rather than assuming 8 hours is enough.

module load cuda120/toolkit/12.0
module load anaconda/3-new

source .venv/bin/activate

python src/training/train_qlora.py \
    --report-to none \
    --output-dir "./models/qlora-adapter" \
    --max-seq-length 192 \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --epochs 2

echo "=== Done. Adapter saved to ./models/qlora-adapter -- pull it back down with scp or rsync. ==="
