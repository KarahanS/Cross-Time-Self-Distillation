@echo off
REM ----------------------------------------------------------------------------
REM  Single-GPU DINO on Windows – VidOR  *single-frame*  baseline
REM  (uses the VidORImageFolder dataset you just added)
REM
REM  ‣ Adjust the two paths below to point at your copy of VidOR
REM    – data_path      : folder that contains  “images/”  and the  *.jsonl  index
REM    – boxes_root     : still required by the arg-parser, but ignored by our loader
REM  ‣ Feel free to tweak hyper-parameters (epochs, lr, etc.) exactly as you would
REM    for ImageNet – nothing else changes.
REM ----------------------------------------------------------------------------

:: ---------------- CUDA / Torch Distributed environment ----------------
set CUDA_VISIBLE_DEVICES=0         
set USE_LIBUV=0                 
set RANK=0
set WORLD_SIZE=1
set LOCAL_RANK=0
set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29655

:: ---------------- Launch DINO (VidOR – single-frame) ----------------
python VIDOR_ODIS/ibot_venice_img.py ^
    --data_path "C:\Users\karab\Desktop\ODIS\venice_sample" ^
    --output_dir ".\debug_vidor_image_dino_run" ^
    --epochs 40 ^
    --batch_size_per_gpu 1 ^
    --num_workers 4 ^
    --arch vit_small ^
    --patch_size 16 ^
    --lr 0.0005 ^
    --local_crops_number 8 ^
    --saveckp_freq 20 ^
    --use_fp16 true ^
    --wandb false ^
    --local_rank 0 ^
    --neighbor_mim_mask false ^
    --static_crop false
