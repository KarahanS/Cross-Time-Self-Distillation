@echo off
REM ===========================================================================
REM  Single-GPU iBOT – WT-Venice  *OBJECT-level*  baseline
REM  (uses ODISWTVeniceFrameDataset inside ibot_venice_obj.py)
REM ===========================================================================

:: ---------------------------------------------------------------------------
:: 0) Distributed / CUDA environment (single machine, 1 GPU)
:: ---------------------------------------------------------------------------
set CUDA_VISIBLE_DEVICES=0
set USE_LIBUV=0
set RANK=0
set WORLD_SIZE=1
set LOCAL_RANK=0
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29655
set TORCH_DISTRIBUTED_DEBUG=DETAIL

:: ---------------------------------------------------------------------------
:: 1) Paths  ——  EDIT THESE TWO LINES
:: ---------------------------------------------------------------------------
set DATA_ROOT=C:\Users\karab\Desktop\ODIS\venice_sample

:: ---------------------------------------------------------------------------
:: 2) Launch iBOT (WT-Venice  object-level)
:: ---------------------------------------------------------------------------
python VIDOR_ODIS/ibot_venice_obj.py ^
    --data_path "%DATA_ROOT%" ^
    --output_dir "%OUT_DIR%" ^
    --local_rank 0 ^
    --epochs 40 ^
    --batch_size_per_gpu 4 ^
    --num_workers 4 ^
    --arch vit_tiny ^
    --patch_size 16 ^
    --num_object_tokens 1 ^
    --global_crops_number 2 ^
    --local_crops_number 0 ^
    --clever_initial_cropping true ^
    --neighbor_mim_mask false ^
    --static_crop false ^
    --lambda1 0.0 ^
    --lambda2 1.0 ^
    --lambda3 1.0 ^
    --lambda_timeimg 0.0 ^
    --lambda_timeobj 1.0 ^
    --saveckp_freq 0 ^
    --resize_first true ^
    --resize_short_side 640

