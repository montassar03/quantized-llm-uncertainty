#!/bin/bash
# LPE-BSR INT-5, 20 members as 4 parallel array tasks (1 GPU each, 5 members each).
# Submit from the repo root:  mkdir -p logs && sbatch scripts/slurm/lpe_bsr_int5.sh
#SBATCH --clusters=hlai
#SBATCH --partition=hlai_std
#SBATCH --job-name=lpe-bsr-int5
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x_%A_%a.out

module load python/3.10.12-base
source ~/venvs/qllm/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd ~/quantized-llm-uncertainty

SEEDS_PER_TASK=5
START=$(( SLURM_ARRAY_TASK_ID * SEEDS_PER_TASK + 1 ))   # task 0: 1-5, task 1: 6-10, ...

python scripts/eval_members.py \
    --model ~/models/Meta-Llama-3-8B-Instruct \
    --mode bsr --bits 5 \
    --seed-start "$START" --num-seeds "$SEEDS_PER_TASK"