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
#SBATCH --output=slurm_outputs/vidor/ibot_%A.out
#SBATCH --error=slurm_outputs/vidor/ibot_%A.err
#SBATCH --array=0
###############################################################################
# 1. Environment
###############################################################################
set -euo pipefail
export WANDB_API_KEY=$(< "${HOME}/.wandb_key")

# 2. Stage the VidOR dataset into fast /tmp on **every** node
###############################################################################
VIDOR_ZIP=/scratch/project_462000938/odis/vidor_60Kclips_5s.zip
TMP_VIDOR=/tmp/vidor                    # will contain  images/   annotations/

LAMBDA1=(0.0 0.0 0.0 0.0 1.0)  # [IMG] level loss
LAMBDA2=(1.0 1.0 1.0 1.0 1.0)  # [PATCH] token loss
LAMBDA3=(1.0 1.0 1.0 1.0 1.0)  # [OBJ] token loss
LAMBDA4=(0.0 0.0 1.0 1.0 1.0)  # [TIME-IMG] loss
LAMBDA5=(1.0 0.0 0.0 1.0 1.0)  # [TIME-OBJ] loss
no_scenarios=5

LOSS_SUFFIX=("Lp_Lo_Lot" "Lp_Lo" "Lo_Lp_Lit" "Lo_Lp_Lit_Lot" "Li_Lo_Lp_Lit_Lot")
lambda1_i=${LAMBDA1[$SLURM_ARRAY_TASK_ID % $no_scenarios]}
lambda2_i=${LAMBDA2[$SLURM_ARRAY_TASK_ID % $no_scenarios]}
lambda3_i=${LAMBDA3[$SLURM_ARRAY_TASK_ID % $no_scenarios]}
lambda4_i=${LAMBDA4[$SLURM_ARRAY_TASK_ID % $no_scenarios]}
lambda5_i=${LAMBDA5[$SLURM_ARRAY_TASK_ID % $no_scenarios]}

no_i=1
loss_suffix_i=${LOSS_SUFFIX[$SLURM_ARRAY_TASK_ID % $no_scenarios]}
echo "lambda1_i: $lambda1_i"
echo "lambda2_i: $lambda2_i"
echo "lambda3_i: $lambda3_i"
echo "lambda4_i (time - [img]): $lambda4_i"
echo "lambda5_i (time - [obj]): $lambda5_i"
echo "no_i: $no_i"
echo "loss_suffix_i: $loss_suffix_i"

NAME=vidor_odis_${loss_suffix_i}_5sec_all2all \


srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} mkdir -p "${TMP_VIDOR}"
srun --ntasks-per-node=1 --nodes=${SLURM_JOB_NUM_NODES} \
     unzip -q "${VIDOR_ZIP}" -d "${TMP_VIDOR}"

# "tg2sg_one2one", "tg2sg_all2all", "sg2th_one2one", "sg2th_all2all", "tg2sg_sg2tg_one2one"
TIME_MATCHING=tg2sg_one2one
OBJ_TIME_MATCHING=tg2sg_all2all

# 3. Run iBOT inside the same container stack you used for HIT
###############################################################################
CKPT_DIR=/scratch/project_462000938/checkpoints
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
     -B "${CODE_DIR}/VIDOR_ODIS:${CODE_DIR}/VIDOR_ODIS" \
     -B "${CKPT_DIR}:${CKPT_DIR}" \
     -B /tmp \
     "${CONTAINER}" \
     "${CODE_DIR}/run.sh" \
       python -u "${CODE_DIR}/VIDOR_ODIS/ibot_vidor_obj.py" \
            --data_path "${TMP_VIDOR}" \
            --boxes_root "${TMP_VIDOR}/annotations" \
            --output_dir "${CKPT_DIR}/${NAME}" \
            --arch vit_small \
            --num_object_tokens $no_i \
            --bb_margin 1 \
            --bb_margin_strategy fixed \
            --object_sampling_strategy random_area \
            --obj_aware_lc_loader false \
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
            --lambda1 $lambda1_i \
            --lambda2 $lambda2_i \
            --lambda3 $lambda3_i \
            --lambda_timeimg $lambda4_i \
            --lambda_timeobj $lambda5_i \
            --time_matching $TIME_MATCHING \
            --obj_time_matching $OBJ_TIME_MATCHING \
            --wandb true \
            --wandb_project vidor \
            --wandb_entity agape \
            --wandb_run_name $NAME \