#!/bin/bash
#SBATCH --account=project_462000938
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8          # 1 task ⇔ 1 GPU
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/eval_grid_search_lp/linprobe.%A.%a.out
#SBATCH --error=slurm_outputs/eval_grid_search_lp/linprobe.%A.%a.err
#SBATCH --array=0                  # ← set to 0-(NUM_CKPTS-1)
# ──────────────────────────────────────────────────────────────────────────
# 1.  CHECKPOINT LIST (add or wildcard as needed)
# ──────────────────────────────────────────────────────────────────────────
CKPTS=(
 # "/scratch/project_462000938/checkpoints/venice_dino_Li_1sec_bs32_FULL/checkpoint0100.pth"
 # "/scratch/project_462000938/checkpoints/venice_dino_Li_Lp_1sec_bs32_FULL/checkpoint0100.pth"
  "/scratch/project_462000938/checkpoints/venice_dino_Li_Lp_Lit_1sec_bs32_FULL/checkpoint0087.pth"
)
NUM_CKPTS=${#CKPTS[@]}

# ──────────────────────────────────────────────────────────────────────────
# 2.  SINGLE TOKEN CHOICE  ("cls"  or  "obj-0")
# ──────────────────────────────────────────────────────────────────────────
TOKEN="cls"          # ← EDIT here

if [[ "$TOKEN" == "cls" ]]; then
  NUM_OBJ_TOK=0
else
  NUM_OBJ_TOK=1      # any "obj-*" token ⇒ 1
fi

# ──────────────────────────────────────────────────────────────────────────
# 3.  Map SLURM_ARRAY_TASK_ID → checkpoint
# ──────────────────────────────────────────────────────────────────────────
if (( SLURM_ARRAY_TASK_ID >= NUM_CKPTS )); then
  echo "[ERROR] Task id ${SLURM_ARRAY_TASK_ID} ≥ NUM_CKPTS ${NUM_CKPTS}" >&2
  exit 1
fi
CKPT_PATH=${CKPTS[$SLURM_ARRAY_TASK_ID]}

echo "────────────────────────────────────────────────────────────"
echo "Array task      : $SLURM_ARRAY_TASK_ID / $((NUM_CKPTS-1))"
echo "Checkpoint      : $CKPT_PATH"
echo "obj_pool token  : $TOKEN   (num_object_tokens = $NUM_OBJ_TOK)"
echo "────────────────────────────────────────────────────────────"

# ──────────────────────────────────────────────────────────────────────────
# 4.  Stage ImageNet once per node (unchanged)
# ──────────────────────────────────────────────────────────────────────────

mkdir -p /tmp/imagenet

if [[ ! -f /tmp/imagenet/completed ]]
then
  if [[ -f /tmp/imagenet/writing ]]
  then
    echo "Other process is writing ImageNet to the disk. Waiting for write to finish."
    while [[ -f /tmp/imagenet/writing ]]
    do
      sleep 60
    done
  else
    echo "Writing ImageNet to /tmp/imagenet"
    touch /tmp/imagenet/writing
    tar --directory=/tmp/imagenet -x -f /scratch/project_462000938/odis/ILSVRC2012_imagefolder.tar
    rm /tmp/imagenet/writing
    touch /tmp/imagenet/completed
  fi
fi

# ──────────────────────────────────────────────────────────────────────────
# 5.  Launch evaluation inside the LUMI container
# ──────────────────────────────────────────────────────────────────────────
CPU_MASKS=0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,\
0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,\
0x000000fe00000000,0x0000fe0000000000


MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1) \
srun --cpu-bind=mask_cpu=$CPU_MASKS \
  singularity exec \
    -B /var/spool/slurmd \
    -B /opt/cray \
    -B /usr/lib64/libcxi.so.1 \
    -B /scratch/project_462000938/linear_probe \
    -B /scratch/project_462000938 \
    -B .:/workdir \
    -B /tmp \
    /scratch/project_462000938/containers/hit_lumi.sif \
    /users/karasari/Object-Level-Self-Supervised-Learning/run.sh \
      python -u /users/karasari/Object-Level-Self-Supervised-Learning/eval_linear_grid_search.py \
        --arch vit_small \
        --output_dir /scratch/project_462000938/linear_probe \
        --pretrained_weights "$CKPT_PATH" \
        --data_path /tmp/imagenet \
        --boxes_root /users/karasari/Object-Level-Self-Supervised-Learning/boxes \
        --num_workers 7 \
        --obj_pool "$TOKEN" \
        --num_object_tokens $NUM_OBJ_TOK
