#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HYDRA_FULL_ERROR=1

# debug for FSDP (Fully Sharded Data Parallel)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# FSDP optimizations
export FSDP_SHARDING_STRATEGY=1  # FULL_SHARD
export FSDP_STATE_DICT_TYPE=SHARDED
export FSDP_OFFLOAD_PARAMS=auto
export FSDP_AUTO_WRAP_POLICY=size_based
export FSDP_BACKWARD_PREFETCH=BACKWARD_POST
export FSDP_CPU_OFFLOAD=true

# huggingface cache dir
export TRANSFORMERS_CACHE="/scratch/jc9723/huggingface"
export HF_HOME="/scratch/jc9723/huggingface"
export HF_DATASETS_CACHE="/scratch/jc9723/huggingface"

timestamp=$(date +"%Y%m%d_%H%M%S")
model="codellama7b"
comment=""

python train.py \
    loss=kto \
    model=${model} \
    datasets=[bigvul] \
    ++exp_name="${timestamp}_${model}_${comment}" \
    ++cache_dir=/scratch/jc9723/huggingface \
