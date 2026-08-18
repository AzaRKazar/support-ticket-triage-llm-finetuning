#!/bin/bash
#SBATCH -J qlora-full-train
#SBATCH -p a100
#SBATCH --gpus=1
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
# 8-hour wall time requested (a100 queue max). If a full 2-epoch run
# looks like it'll need longer based on the timing check's per-step
# rate, switch -p to a100-long (up to 48 hours, 1 node) before
# submitting.

module load cuda120/toolkit/12.0
module load anaconda3 2>/dev/null || true

source .venv/bin/activate

python src/training/train_qlora.py \
    --report-to none \
    --output-dir "./models/qlora-adapter" \
    --max-seq-length 192 \
    --batch-size 16 \
    --grad-accum-steps 2 \
    --epochs 2

echo "=== Done. Adapter saved to ./models/qlora-adapter -- pull it back down with scp or rsync. ==="
