#!/bin/bash
#SBATCH -p <dummy_name>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:Tesla-V100-32GB:4
#SBATCH --cpus-per-task=2
#SBATCH --mem=60G
#SBATCH --job-name=knee_eval_ad_ax0
#SBATCH --output=slurm.out

eval "$(conda shell.bash hook)"
conda activate pytorch

# python -m torch.distributed.launch --nproc_per_node=4 main.py 
python -m torch.distributed.torchrun --nproc_per_node=4 main.py 


