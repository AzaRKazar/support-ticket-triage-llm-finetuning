#!/bin/bash
#SBATCH -J qlora-timing-check
#SBATCH -p v100
#SBATCH --gres=gpu:1
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
# Verified live on SeaWulf 2026-08-18: no `a100` partition exists on this
# cluster (public docs were wrong). Real GPU partitions are `gpu`/
# `gpu-long`/`gpu-large` (Tesla K80 -- no tensor cores, likely slower than
# Colab's T4, avoid) and `v100` (Tesla V100-PCIE-32GB, 24hr limit -- used
# here). `nvidia-smi -L` via `srun` is what confirmed the actual hardware,
# not the node/partition names.

module load cuda120/toolkit/12.0
module load anaconda/3-new

# --system-site-packages inherits anaconda/3-new's precompiled numpy/pandas,
# but pip's build isolation (used when compiling packages from source)
# ignores it and fetches its own numpy anyway -- so scikit-learn/scipy
# still tried and failed to build numpy against this node's GCC 4.8.5
# (needs >=9.3). Fix: install only what train_qlora.py actually imports
# (requirements-seawulf.txt), skipping scikit-learn/scipy/matplotlib/
# seaborn entirely since training doesn't need them.
if [ ! -d ".venv" ]; then
    python -m venv --system-site-packages .venv
fi
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements-seawulf.txt
pip install -q torch --index-url https://download.pytorch.org/whl/cu121
# Known issue: pip can resolve nvidia-cusparse-cu12/nvidia-nvjitlink-cu12
# (loose-range sub-deps of the cu121 torch wheel) to a mismatched pair,
# causing "symbol __nvJitLinkAddData_12_1 ... not defined" on import.
# Forcing nvjitlink to the latest available version is the documented fix.
pip install -q --upgrade nvidia-nvjitlink-cu12

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
