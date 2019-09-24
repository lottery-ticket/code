#!/usr/bin/env bash

LTH_DIR="$(dirname $0)/.."

source "$(dirname $0)/common.sh"

if [ "$#" -lt 1 ]; then
    echo "Must provide an experiment name!"
    exit 1
fi

EXPERIMENT_NAME="$1"; shift

MODEL_DIR="${FS_PREFIX}/execution_data/${EXPERIMENT_NAME}";
RESULTS_DIR="${FS_PREFIX}/results/${EXPERIMENT_NAME}";
BENCHMARK_DIR="${FS_PREFIX}/benchmark_logs/${EXPERIMENT_NAME}";

cd "${LTH_DIR}/gpu-src"

run_or_debug python "official/resnet/cifar10_main.py" \
       --data_dir "${FS_PREFIX}/cifar_binary" \
       --resnet_size "${RESNET_SIZE}" \
       --batch_size 128 \
       --datasets_num_parallel_batches 1 \
       --benchmark_logger_type BenchmarkFileLogger \
       --benchmark_log_dir "${BENCHMARK_DIR}" \
       --model_dir "${MODEL_DIR}" \
       --lottery_results_dir "${RESULTS_DIR}" \
       --lottery_checkpoint_iters 0,14,404,795,1576,4701,8608,12514,16420,20326,24233,28139,32045,35951,39858,43764,47670,51576,55483,59389,63295,67201,69545,70326,70717,71108 \
       "${@}"
