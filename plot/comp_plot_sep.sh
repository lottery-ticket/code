#!/usr/bin/env bash

for i in "${@}"; do
    n_epochs=$(python -c "print(int(182 - round($i / 391.0)))")
    args=()
    args+=("gs://REDACTED/results/resnet20/global_magnitude_20/v2/lottery/prune_${i}")
    args+=("gs://REDACTED/results/resnet20/global_magnitude_20/v2/finetune/finetune_${n_epochs}")
    "$(dirname $0)"/experiments.py "${args[@]}" &
done

wait
