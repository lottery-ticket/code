#!/usr/bin/env bash

# set -eux

function enqueue() {
    echo "${@}"
    # "$(dirname $0)/../enqueue.sh" "${@}"
    # sleep 1
}

for i in 0 11259 22518 33777 45036 56295 67554 78813 90072 101331 112590; do
    for j in 1 2 3; do
        TRIAL_DIR="resnet50/prune_global_20/v10/finetune/finetune_9/trial_${j}/iter_0"
        for DENSITY in 80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74; do
            enqueue /home/lth/lth/resnet50_expts/oneshot_lottery.sh "$i" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet50_expts/oneshot_finetune.sh "$(expr \( 112590 - $i \) / 1251 )" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet50_expts/oneshot_reinit.sh "$i" "$j" "${TRIAL_DIR}" "${DENSITY}"
        done
    done
done
