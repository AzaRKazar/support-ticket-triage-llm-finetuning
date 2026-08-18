#!/bin/bash
#SBATCH -J qlora-timing-check
#SBATCH -p a100
#SBATCH --gpus=1
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -o slurm-timing-%j.out
#SBATCH -e slurm-timing-%j.err

# Step 1 of 2: bounded timing check only. Submit this first, inspect the
# step times in slurm-timing-<jobid>.out, and only submit
# seawulf_full_train.sbatch afterward if they look stable (not climbing
# the way they did on the local laptop GPU). Same discipline as the
# local and Colab attempts -- don't commit to a multi-hour job blind.
#
# CHECK BEFORE FIRST RUN: `module avail` on SeaWulf to confirm the module
# names below still match -- pulled from public SeaWulf docs, not
# verified against a live session.

module load cuda120/toolkit/12.0
module load anaconda3 2>/dev/null || true

if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate

pip install -q -r requirements.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
pip install -q bitsandbytes peft trl

echo "=== GPU check ==="
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== Timing check: 20 bounded steps ==="
python src/training/train_qlora.py \
    --max-steps 20 \
    --report-to none \
    --output-dir "./models/timing-test" \
    --max-seq-length 192 \
    --batch-size 16

echo "=== Timing check complete. Inspect step times above before submitting the full run. ==="
