#!/bin/bash
#SBATCH --job-name=ibot_vidor_img
#SBATCH --account=project_462000938
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8            # 1 task  ⇔  1 GPU
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/vidor_v2/ibot_%A.out
#SBATCH --error=slurm_outputs/vidor_v2/ibot_%A.err
###############################################################################
# 1. Environment
###############################################################################
set -euo pipefail
export WANDB_API_KEY=$(< "${HOME}/.wandb_key")

# 2. Stage the VidOR dataset into fast /tmp on **every** node
###############################################################################
VIDOR_ZIP=/scratch/project_462000938/odis/vidor_60Kclips_5s.zip
TMP_VIDOR=/tmp/vidor                    # will contain  images/   annotations/

srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} mkdir -p "${TMP_VIDOR}"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
     unzip -q "${VIDOR_ZIP}" -d "${TMP_VIDOR}"

# "tg2sg_one2one", "tg2sg_all2all", "sg2th_one2one", "sg2th_all2all", "tg2sg_sg2tg_one2one"
TIME_MATCHING=tg2sg_one2one


NAME=vidor_5sec_Li_Lp_Lit_Lpt_neighmask-shared
# 3. Run iBOT inside the same container stack you used for HIT
###############################################################################
CKPT_DIR=/scratch/project_462000938/checkpoints/${NAME}
CONTAINER=/scratch/project_462000938/containers/hit_lumi.sif  
CODE_DIR=/users/karasari/Object-Level-Self-Supervised-Learning

mkdir -p "${CKPT_DIR}"

# CPU-affinity mask (copy-paste from your previous script)
CPU_MASKS=0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000

MASTER_ADDR=$(scontrol show hostname "${SLURM_NODELIST}" | head -n1) \
srun --cpu-bind=mask_cpu=${CPU_MASKS} \
     singularity exec \
     -B /var/spool/slurmd \
     -B /opt/cray \
     -B /usr/lib64/libcxi.so.1 \
     -B "${CODE_DIR}:${CODE_DIR}" \
     -B "${CKPT_DIR}:${CKPT_DIR}" \
     -B "${CODE_DIR}/VIDOR_ODIS:${CODE_DIR}/VIDOR_ODIS" \
     -B /tmp \
     "${CONTAINER}" \
     "${CODE_DIR}/run.sh" \
       python -u "${CODE_DIR}/VIDOR_ODIS/ibot_vidor_img.py" \
            --dataset VidORiBOTFrameDataset \
            --data_path "${TMP_VIDOR}" \
            --boxes_root "${TMP_VIDOR}/annotations" \
            --output_dir "${CKPT_DIR}" \
            --arch vit_small \
            --teacher_temp 0.07 \
            --warmup_teacher_temp 0.04 \
            --warmup_teacher_temp_epochs 30 \
            --warmup_teacher_patch_temp 0.04 \
            --teacher_patch_temp 0.07 \
            --norm_last_layer false \
            --epochs 300 \
            --batch_size_per_gpu 32 \
            --out_dim 8192 \
            --local_crops_number 10 \
            --global_crops_scale 0.25 1 \
            --local_crops_scale 0.05 0.25 \
            --pred_ratio 0 0.3 \
            --pred_ratio_var 0 0.2 \
            --shared_head true \
            --shared_head_teacher true \
            --use_fp16 false \
            --num_workers 7 \
            --saveckp_freq 50 \
            --lambda1 1.0 \
            --lambda2 1.0 \
            --lambda_timeimg 1.0 \
            --lambda_timepatch 1.0 \
            --time_matching $TIME_MATCHING \
            --neigh_mim "shared" \
            --wandb true \
            --wandb_project vidor \
            --wandb_entity agape \
            --wandb_run_name ${NAME} \

