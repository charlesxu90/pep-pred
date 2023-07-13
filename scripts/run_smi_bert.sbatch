#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --mem=64G
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --partition=batch
#SBATCH --output=log-%x-slurm-%j.out
#SBATCH --error=log-%x-slurm-%j.err
#SBATCH --reservation=A100

module load gcc/11.1.0

# torchrun --nproc_per_node=4 train_bert.py --config configs/train_smi_bert.yaml --output_dir results/train_smi_bert
torchrun --nproc_per_node=4  train_bert.py --config configs/train_smi_bert_tune.yaml --output_dir results/train_smi_bert_tune --ckpt results/train_smi_bert/model_3_0.277.pt
