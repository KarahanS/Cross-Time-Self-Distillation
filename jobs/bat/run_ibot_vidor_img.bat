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
:: 2) Launch iBOT (WT-Venice  object-level)
:: ---------------------------------------------------------------------------
python VIDOR_ODIS/ibot_vidor_img_with_memory.py ^
    --data_path "C:\Users\karab\Desktop\ODIS\vidor_sample" ^
    --boxes_root "C:\Users\karab\Desktop\ODIS\vidor_sample\annotations" ^
    --output_dir ".\debug_vidor_image_dino_run" ^
    --local_rank 0 ^
    --epochs 40 ^
    --batch_size_per_gpu 4 ^
    --num_workers 4 ^
    --arch vit_tiny ^
    --patch_size 16 ^
    --num_object_tokens 1 ^
    --global_crops_number 2 ^
    --local_crops_number 0 ^
    --lambda1 0.0 ^
    --lambda2 1.0 ^
    --lambda_timeimg 0.0 ^
    --lambda_timepatch 1.0 ^
    --neigh_mim "random" ^
    --saveckp_freq 0

