#!/bin/bash
#SBATCH --job-name=gr00t-behavior-ft
#SBATCH --output=logs/gr00t_behavior_ft_%j.out
#SBATCH --error=logs/gr00t_behavior_ft_%j.err
#SBATCH --partition=A100-80GB,H200
#SBATCH --qos=hpgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n80
# --------------------------- ENVIRONMENT SETUP -------------------------- #
source ~/.bashrc
export PYTHONPATH=/home/khj20343/Postech-Behavior-Challenge/b1k-baselines/baselines/Isaac-GR00T:${PYTHONPATH}
conda activate gr00t
set -u

export HF_HOME=${HF_HOME:-/home/khj20343/.cache/huggingface}
export WANDB_ENTITY=${WANDB_ENTITY:-goodi20343-korea-advanced-institute-of-science-and-techn}
export WANDB_PROJECT=${WANDB_PROJECT:-behavior-1k}
export MASTER_PORT=${MASTER_PORT:-29521}

mkdir -p logs

# Determine GPU world size (defaults to the number of GPUs allocated by Slurm)
NUM_GPUS=${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}

# --------------------------- TRAINING COMMAND --------------------------- #
DATASET_DIR=${DATASET_DIR:-/home/khj20343/Postech-Behavior-Challenge/datasets/2025-challenge-demos}
OUTPUT_DIR=${OUTPUT_DIR:-/home/khj20343/Postech-Behavior-Challenge/runs/gr00t_behavior_head_wrist_rgb}
BATCH_SIZE=${BATCH_SIZE:-48}
MAX_STEPS=${MAX_STEPS:-100000}
SAVE_STEPS=${SAVE_STEPS:-1000}
DATA_CONFIG=${DATA_CONFIG:-examples.Behavior.custom_data_config:BehaviorDataConfig}
TASKS=${TASKS:-}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}

mkdir -p "${OUTPUT_DIR}"

CMD=(python /home/khj20343/Postech-Behavior-Challenge/b1k-baselines/baselines/Isaac-GR00T/scripts/gr00t_behavior_finetune.py)
CMD+=(--dataset-path "${DATASET_DIR}")
CMD+=(--data-config "${DATA_CONFIG}")
if [[ -n "${TASKS}" ]]; then
  for task in ${TASKS}; do
    CMD+=(--tasks "${task}")
  done
fi
CMD+=(--output-dir "${OUTPUT_DIR}")
CMD+=(--batch-size "${BATCH_SIZE}")
CMD+=(--max-steps "${MAX_STEPS}")
CMD+=(--save-steps "${SAVE_STEPS}")
CMD+=(--num-gpus "${NUM_GPUS}")
CMD+=(--dataloader-num-workers "${DATALOADER_NUM_WORKERS}")

"${CMD[@]}"

echo "GR00T Behavior fine-tuning job completed."

