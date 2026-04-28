#!/bin/bash
#SBATCH --job-name=train_nano_gpt
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1    #1 process per node
#SBATCH --cpus-per-task=15     #15 CPU cores each
#SBATCH --time=00:30:00         #30 minutes
#SBATCH --output=train_nano_gpt.%j.out
#SBATCH --error=train_nano_gpt.%j.err

set -euo pipefail

#environment setup
module load miniconda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ~/myenv

# If you don't have infiniband, keep it off
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1

# use all the threads we requested for PyTorch's internal pools
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# pick master first hostname in the allocation
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR MASTER_PORT: $MASTER_PORT"
echo "Nodes: $SLURM_JOB_NODELIST"

srun bash -lc '
    export RANK=$SLURM_PROCID
    export WORLD_SIZE=$SLURM_NTASKS
    export LOCAL_RANK=$SLURM_LOCALID
    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    echo "[$(hostname)] RANK=$RANK WORLD_SIZE=$WORLD_SIZE LOCAL_RANK=$LOCAL_RANK 
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=$OMP_NUM_THREADS"
    python -u train.py config/train_shakespeare_char.py
'