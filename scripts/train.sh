#!/bin/bash

# Default values
NUM_GPUS=8
MODEL_SIZE="base"
MAX_LENGTH=8192
BATCH_SIZE=32
GRAD_ACCUM=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --model)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${MODEL_SIZE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Launch training
echo "Starting training with configuration:"
echo "Model: ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Max Length: ${MAX_LENGTH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation: ${GRAD_ACCUM}"
echo "Output Directory: ${OUTPUT_DIR}"

torchrun \
    --nproc_per_node="$NUM_GPUS" \
    --master_port=$(( RANDOM % 50000 + 10000 )) \
    training/trainer.py \
    model="$MODEL_SIZE" \
    data.max_length="$MAX_LENGTH" \
    training.batch_size="$BATCH_SIZE" \
    training.gradient_accumulation_steps="$GRAD_ACCUM" \
    output_dir="$OUTPUT_DIR" \
    2>&1 | tee "$OUTPUT_DIR/training.log" 