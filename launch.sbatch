#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=800GB
#SBATCH --cpus-per-task=4

#SBATCH --time=1-23:59:59
#SBATCH --output=var/logs/%A/%a.out
#SBATCH --error=var/logs/%A/%a.err
#SBATCH --account=pr_177_general
#SBATCH --mail-type=ALL
#SBATCH --mail-user=junyao.chen@nyu.edu

# --job-name comes from the run_batch wrapper script
nvidia-smi

export PATH="/scratch/jc9723/miniconda3/bin:$PATH"
conda init
eval "$(/scratch/jc9723/miniconda3/bin/conda shell.bash hook)"
echo "[BEFORE] Conda env: $CONDA_DEFAULT_ENV"
echo "[BEFORE] Conda prefix: $CONDA_PREFIX"
source activate code-sec


echo "Python: $(which python)"
echo "[AFTER] Conda env: $CONDA_DEFAULT_ENV"
echo "[AFTER] Conda prefix: $CONDA_PREFIX"
echo "Python path:"
python -c "import sys; print('\n'.join(sys.path))"
echo "Installed packages:"
conda list




cd /scratch/jc9723/code-sec
pwd
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"

./train.sh