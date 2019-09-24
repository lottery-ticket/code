#!/usr/bin/env bash

args=()
for i in "${@}"; do
    n_epochs=$(python -c "print(int(182 - round($i / 391.0)))")
    args+=("gs://REDACTED/results/resnet20/global_magnitude_20/v2/lottery/prune_${i}")
    args+=("gs://REDACTED/results/resnet20/global_magnitude_20/v2/finetune/finetune_${n_epochs}")
done

"$(dirname $0)"/iter_tradeoff.py --iter 1 "${args[@]}"
