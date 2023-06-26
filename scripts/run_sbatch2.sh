#!/bin/bash

#SBATCH --job-name=COOL_JOB_NAME    # create a short name for your job
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=240GB               # total memory per node 
#SBATCH --time=23:59:00          # total run time limit (HH:MM:SS)




##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
source ~/.bashrc


# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
# ******************************************************************************************

# zoom zoom - recommended from lightning
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32


echo "Run started at:- "
date

# Actual run of script 
#srun python main.py # Use this if you have python in your environment
srun singularity exec --nv /Location/Of/Your/Containers/pytorch_22.05-py3.sif python main.py
echo ""
echo "################################################################"
echo "@@@@@@@@@@@@@@@@@@ Run completed at:- @@@@@@@@@@@@@@@@@@@@@@@@@"
date
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "################################################################"

