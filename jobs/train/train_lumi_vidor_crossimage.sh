#!/bin/bash
###############################################################################
#  LUMI - iBOT / VidOR   –  CLS + PATCH + CROSS-IMG losses
###############################################################################
#SBATCH --job-name=ibot_vidor_imgPatchXimg
#SBATCH --account=project_462000938
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8          # 16 GPUs total
#SBATCH --ntasks-per-node=8        # one MPI rank per GPU
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/train_ibot_vidor_ximage/ibot_%A.out
#SBATCH --error=slurm_outputs/train_ibot_vidor_ximage/ibot_%A.err
###############################################################################
# 1. House-keeping & env
###############################################################################
set -euo pipefail
export WANDB_API_KEY="$(< "${HOME}/.wandb_key")"

###############################################################################
# 2. Stage VidOR (unzipped once per node → /tmp, the fast NVMe SSD)
###############################################################################
VIDOR_ZIP=/scratch/project_462000938/odis/vidor_63K_clips.zip
TMP_VIDOR=/tmp/vidor                         # will contain  images/  annotations/

echo "Dataset: ${VIDOR_ZIP}"

srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
     mkdir -p "${TMP_VIDOR}"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
     unzip -q "${VIDOR_ZIP}" -d "${TMP_VIDOR}"

###############################################################################
# 3. Paths – change only if you moved things
###############################################################################
CONTAINER=/scratch/project_462000938/containers/hit_lumi.sif
CODE_DIR=/users/karasari/Object-Level-Self-Supervised-Learning
SCRIPT=${CODE_DIR}/ibot_vidor.py        
CKPT_DIR=/scratch/project_462000938/checkpoints/ibot_vidor63K_video_Li_Lp_Lit

mkdir -p "${CKPT_DIR}"

###############################################################################
# 4. Affinity mask (copy-paste from an earlier good run)
###############################################################################
CPU_MASKS=0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000

###############################################################################
# 5. Launch (1 MPI rank ⇔ 1 GPU)
###############################################################################
MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1) \
srun --cpu-bind=mask_cpu=${CPU_MASKS} \
     singularity exec \
       -B /var/spool/slurmd \
       -B /opt/cray \
       -B /usr/lib64/libcxi.so.1 \
       -B "${CODE_DIR}:${CODE_DIR}" \
       -B "${CKPT_DIR}:${CKPT_DIR}" \
       -B /tmp \
       "${CONTAINER}" \
       "${CODE_DIR}/run.sh" \
         python -u "${SCRIPT}" \
           --arch vit_small \
           --data_path       "${TMP_VIDOR}" \
           --boxes_root      "${TMP_VIDOR}/annotations" \
           --output_dir      "${CKPT_DIR}" \
           --dataset         "VidORPrebuiltClipDataset" \
           --epochs          300 \
           --batch_size_per_gpu 32 \
           --num_workers     7 \
           --global_crops_number 2 \
           --local_crops_number 10 \
           --global_crops_scale 0.25 1 \
           --local_crops_scale 0.05 0.25 \
           --pred_ratio 0 0.3 \
           --pred_ratio_var 0 0.2 \
           --out_dim 8192 \
           --shared_head true \
           --shared_head_teacher true \
           --use_fp16 false \
           --saveckp_freq 50 \
           --seed 0 \
           --lambda1        1.0 \
           --lambda2        1.0 \
           --num_object_tokens 0 \
           --lambda_timeimg 1.0 \
           --pair_sampling  all \
           --time_matching  one2one \
           --wandb          true \
           --wandb_project  vidor \
           --wandb_entity   agape \
           --wandb_run_name ibot_vidor63K_video_Li_Lp_Lit
