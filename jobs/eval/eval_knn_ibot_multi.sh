#!/bin/bash
#SBATCH --job-name=knn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-h200-141g-short,gpu-h100-80g-short,gpu-h100-80g,gpu-v100-32g,gpu-a100-80g,gpu-v100-16g
#SBATCH --mem=32GB
#SBATCH --output=slurm_outputs/knn_ibot_cls_true/knn-gpu.%A.%a.out
#SBATCH --error=slurm_outputs/knn_ibot_cls_true/knn-gpu.%A.%a.err
#SBATCH --tmp=200G
#SBATCH --array=0-1
# ────────────────────────────────────────────────────────────────────────────
# 1. EXPERIMENT CONFIGS  (3 total)
#    Format  "<knn_token> <use_bounding_boxes>"
# ────────────────────────────────────────────────────────────────────────────


# this is cls true,
# if you want cls false, call odis_multi.sh
CONFIGS=(
  "cls" 
)


NUM_CONFIGS=${#CONFIGS[@]}

# ────────────────────────────────────────────────────────────────────────────
# 2. CHECKPOINT LIST  (wildcard expansion happens at runtime)
# ────────────────────────────────────────────────────────────────────────────


CKPTS=(
 "/scratch/work/saritak1/checkpoints/ibot_ximgobj_fp16_adasim/checkpoint.pth"
  "/scratch/work/saritak1/checkpoints/ibot-coco-300e/checkpoint.pth"
 #"/scratch/work/saritak1/checkpoints/ibot+cribo_fp16_Li_Lp_Lnn_Lc/checkpoint.pth"
#"/scratch/work/saritak1/checkpoints/venice_dino_Li_1sec_bs32_FULL/checkpoint0100.pth"
#"/scratch/work/saritak1/checkpoints/venice_dino_Li_Lp_1sec_bs32_FULL/checkpoint0100.pth"
#"/scratch/work/saritak1/checkpoints/venice_dino_Li_Lp_Lit_1sec_bs32_FULL/checkpoint0087.pth"

#/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_1sec_instance_bs32/checkpoint0250.pth
#/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_3sec_instance_bs32/checkpoint0250.pth


#"/scratch/work/saritak1/checkpoints/ibot_Li_Lp/checkpoint.pth"
#"/scratch/work/saritak1/checkpoints/ibot-coco-300e/checkpoint.pth"
#"/scratch/work/saritak1/checkpoints/ibot_cribo_Li_Lp_Lnn_Lc/checkpoint.pth"
#"/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_5sec_bs32_withsharedMIMMask/checkpoint0250.pth"
#"/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_10sec_bs32_withsharedMIMMask/checkpoint0250.pth"
#"/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_15sec_bs32_withsharedMIMMask/checkpoint0250.pth"
#"/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_20sec_bs32_withsharedMIMMask/checkpoint0250.pth"
#"/scratch/work/saritak1/checkpoints/vidor_odis_Lp_Lo_Lot_25sec_bs32_withsharedMIMMask/checkpoint0250.pth"

)


NUM_CKPTS=${#CKPTS[@]}

if (( NUM_CKPTS == 0 )); then
  echo "No checkpoints found — pattern might be wrong." >&2
  exit 1
fi

TOTAL_TASKS=$(( NUM_CKPTS * NUM_CONFIGS ))


if (( SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
  echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID} exceeds total tasks ${TOTAL_TASKS}." >&2
  exit 1
fi

CKPT_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_CONFIGS ))
CONFIG_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_CONFIGS ))

CKPT_PATH=${CKPTS[$CKPT_INDEX]}
read knn_token_i use_boxes_i <<< "${CONFIGS[$CONFIG_INDEX]}"

echo "────────────────────────────────────────────────────────"
echo "Array task      : $SLURM_ARRAY_TASK_ID / $((TOTAL_TASKS-1))"
echo "Checkpoint      : $CKPT_PATH"
echo "KNN token       : $knn_token_i"
echo "Use bboxes flag : $use_boxes_i"
echo "────────────────────────────────────────────────────────"

# ────────────────────────────────────────────────────────────────────────────
# 4. ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────f───
module load mamba
eval "$(conda shell.bash hook)"
conda activate "/scratch/work/saritak1/conda/miniconda3/envs/odis"

# ────────────────────────────────────────────────────────────────────────────
# 5. OPTIONAL: Stage ImageNet once per node (kept here unchanged)
# ────────────────────────────────────────────────────────────────────────────
mkdir -p /tmp/imagenet
if [[ ! -f /tmp/imagenet/completed ]]; then
  if [[ -f /tmp/imagenet/writing ]]; then
    echo "Another task is writing ImageNet → waiting ..."
    while [[ -f /tmp/imagenet/writing ]]; do sleep 60; done
  else
    echo "Unpacking ImageNet to /tmp/imagenet ..."
    touch /tmp/imagenet/writing
    tar -C /tmp/imagenet -xf /scratch/shareddata/dldata/imagenet_imagefolder/ILSVRC2012_imagefolder.tar
    rm  /tmp/imagenet/writing
    touch /tmp/imagenet/completed
  fi
fi

# ────────────────────────────────────────────────────────────────────────────
# 6. RUN
# ────────────────────────────────────────────────────────────────────────────
CODE_DIR=/home/saritak1/Object-Level-Self-Supervised-Learning
PORT=$(( 29650 + SLURM_ARRAY_TASK_ID ))

echo "Torchrun port   : $PORT"

torchrun --nproc_per_node=1 --rdzv_endpoint=localhost:$PORT \
  "${CODE_DIR}/eval_knn_ibot.py"                                 \
    --pretrained_weights  "$CKPT_PATH"                      \
    --checkpoint_key      teacher                           \
    --data_path           /tmp/imagenet                     \
    --boxes_root          "${CODE_DIR}/boxes"               \
    --arch                "vit_small"                        \
    --num_workers         8
