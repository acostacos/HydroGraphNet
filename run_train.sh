#!/bin/sh
#SBATCH --job-name=hydrographnet_train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-80:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

export LOCAL_CACHE="~/.cache/physicsnemo/"
export HGN_BASE_DIR="~/HydroGraphNet"
export PYTHONPATH="${PYTHONPATH}:~/HydroGraphNet"

. venv/bin/activate

srun python examples/weather/flood_modeling/hydrographnet/train.py
