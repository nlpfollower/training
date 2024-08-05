#!/bin/bash
export MASTER_ADDR=$1
export MASTER_PORT=$2
export WORLD_SIZE=$3
export NODE_RANK=$4
export NPROC_PER_NODE=$5
export MODEL_TYPE=$6

python train.py --nnodes $WORLD_SIZE --node_rank $NODE_RANK --nproc_per_node $NPROC_PER_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT --model $MODEL_TYPE