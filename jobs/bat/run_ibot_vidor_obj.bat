@echo off
REM ---------------------------------------------------------------------------
REM  Single-GPU iBOT on Windows – VidOR  *IMAGE*  baseline
REM  (uses VidORImageMask dataset, one frame per “clip”)
REM
REM ---------------------------------------------------------------------------

:: ---------------- Distributed/CUDA environment ----------------
set CUDA_VISIBLE_DEVICES=0      
set USE_LIBUV=0                 
set RANK=0
set WORLD_SIZE=1
set LOCAL_RANK=0
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29655
set TORCH_DISTRIBUTED_DEBUG=DETAIL

:: ---------------- Launch iBOT (VidOR - image mode) ----------------
python VIDOR_ODIS\ibot_vidor_obj.py ^
    --data_path "C:\Users\karab\Desktop\ODIS\vidor_sample" ^
    --boxes_root "C:\Users\karab\Desktop\ODIS\vidor_sample\annotations" ^
    --output_dir ".\debug_vidor_image_ibot_run" ^
    --local_rank 0 ^
    --saveckp_freq 0 ^
    --epochs 40 ^
    --batch_size_per_gpu 4 ^
    --num_workers 4 ^
    --arch vit_tiny ^
    --patch_size 16 ^
    --num_object_tokens 1 ^
    --global_crops_number 2 ^
    --local_crops_number 8 ^
    --lambda1 0.0 ^
    --lambda2 1.0 ^
    --lambda3 1.0 ^
    --lambda_timeimg 0.0 ^
    --lambda_timeobj 1.0 ^