#!/bin/bash

# Default values
NUM_GPUS=1
MODEL_SIZE="base"
BATCH_SIZES=(1 2 4 8 16 32)
SEQ_LENGTHS=(1024 2048 4096 8192 16384 32768)
WARMUP_STEPS=10
BENCHMARK_STEPS=50

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
        --warmup)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --steps)
            BENCHMARK_STEPS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=OFF

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmarks/${MODEL_SIZE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Function to run benchmark
run_benchmark() {
    local batch_size=$1
    local seq_length=$2
    local output_file="$OUTPUT_DIR/benchmark_b${batch_size}_s${seq_length}.log"
    
    echo "Running benchmark with batch_size=$batch_size, seq_length=$seq_length"
    
    python training/benchmark.py \
        model="$MODEL_SIZE" \
        training.batch_size="$batch_size" \
        data.max_length="$seq_length" \
        profiling.enabled=true \
        profiling.warmup_steps="$WARMUP_STEPS" \
        profiling.benchmark_steps="$BENCHMARK_STEPS" \
        2>&1 | tee "$output_file"
}

# Run benchmarks
echo "Starting benchmarks with configuration:"
echo "Model: ${MODEL_SIZE}"
echo "GPUs: ${NUM_GPUS}"
echo "Warmup Steps: ${WARMUP_STEPS}"
echo "Benchmark Steps: ${BENCHMARK_STEPS}"
echo "Output Directory: ${OUTPUT_DIR}"

# Run for different batch sizes and sequence lengths
for batch_size in "${BATCH_SIZES[@]}"; do
    for seq_length in "${SEQ_LENGTHS[@]}"; do
        run_benchmark "$batch_size" "$seq_length"
    done
done

# Generate summary report
python scripts/summarize_benchmarks.py \
    --input-dir "$OUTPUT_DIR" \
    --output-file "$OUTPUT_DIR/summary.csv"

# Plot results
python scripts/plot_benchmarks.py \
    --input-file "$OUTPUT_DIR/summary.csv" \
    --output-dir "$OUTPUT_DIR/plots" 