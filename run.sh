#!/bin/bash -e

ls /
ls /tmp

# Report affinity
echo "Rank $SLURM_PROCID --> $(taskset -p $$)"

# Report GPUs
if [ $SLURM_LOCALID -eq 0 ] ; then
    rocm-smi
else
  sleep 2
fi

# Start conda environment inside the container
# $WITH_CONDA

# Setting the caches relevant to our application.
export TORCH_HOME=/users/karasari/Object-Level-Self-Supervised-Learning/torch-cache
export HF_HOME=/users/karasari/Object-Level-Self-Supervised-Learning/hf-cache
export TOKENIZERS_PARALLELISM=false

# Tell RCCL to use only Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# Tell MIOpen where to store its cache
export MIOPEN_USER_DB_PATH="/tmp/karasari-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH

if [ $SLURM_LOCALID -eq 0 ] ; then
  rm -rf $MIOPEN_USER_DB_PATH
  mkdir -p $MIOPEN_USER_DB_PATH    
else
  sleep 2
fi

# export NCCL_DEBUG=INFO 
# export NCCL_DEBUG_SUBSYS=INIT,COLL
# export NCCL_DEBUG_FILE=/tmp/hizlicag-rccl-rank$SLURM_PROCID.txt

# Translate SLURM environment 

export MASTER_PORT=25900
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=8
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

set -x

# Run application
eval "$@"

