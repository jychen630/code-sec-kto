#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_VISIBLE_DEVICES=4,7
export HYDRA_FULL_ERROR=1

# debug for FSDP (Fully Sharded Data Parallel)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# huggingface cache dir
export TRANSFORMERS_CACHE="/local/nlp/junyao/huggingface"
export HF_HOME="/local/nlp/junyao/huggingface"
export HF_DATASETS_CACHE="/local/nlp/junyao/huggingface"

python train.py \
    loss=kto \
    model=starcoder \
    datasets=[bigvul] \
    exp_name=first_test \
    ++cache_dir=/local/nlp/junyao/huggingface \
