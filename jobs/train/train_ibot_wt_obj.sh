#!/bin/bash
#SBATCH --job-name=ibot_vidor_img
#SBATCH --account=<your_project_id_here>
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8            # 1 task  ⇔  1 GPU
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --threads-per-core=1
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=slurm_outputs/train_ibot_venice/ibot_%A.out
#SBATCH --error=slurm_outputs/train_ibot_venice/ibot_%A.err

###############################################################################
# 1. Environment
###############################################################################
set -euo pipefail
export WANDB_API_KEY=$(< "${HOME}/.wandb_key")

# 2. Stage the VidOR dataset into fast /tmp on **every** node
###############################################################################
VENICE_ZIP=/scratch/<project_id>/wt_venice/venice_1sec.zip
TMP_VENICE=/tmp/venice                    # will contain  images/   annotations/

srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} mkdir -p "${TMP_VENICE}"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
     unzip -q "${VENICE_ZIP}" -d "${TMP_VENICE}"

# "tg2sg_one2one", "tg2sg_all2all", "sg2th_one2one", "sg2th_all2all", "tg2sg_sg2tg_one2one"
TIME_MATCHING=tg2sg_one2one


NAME=Lo_Lp_venice_1sec_FULL
# 3. Run iBOT inside the same container stack you used for HIT
###############################################################################
CKPT_DIR=/scratch/<project_id>/checkpoints/${NAME}
CONTAINER=/scratch/<project_id>/containers/<container>.sif
CODE_DIR=/users/<username>/<repo_name>

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
       python -u "${CODE_DIR}/VIDOR_ODIS/ibot_venice_obj.py" \
            --data_path "${TMP_VENICE}" \
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
            --lambda1 0.0 \
            --lambda2 1.0 \
            --lambda3 1.0 \
            --lambda_timeimg 0.0 \
            --lambda_timeobj 0.0 \
            --time_matching $TIME_MATCHING \
            --wandb true \
            --wandb_project vidor \
            --wandb_entity <wandb_entity> \
            --wandb_run_name ${NAME} \
            --neighbor_mim_mask false \
            --static_crop false \
            --resize_first true \
            --resize_short_side 640 \
            --num_object_tokens 1
