#!/usr/bin/env bash

DENSITIES=(80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74)
for j in 1 2 3; do
    for it in `seq 0 9`; do
        next="$(expr ${it} + 1)"
        DENSITY="${DENSITIES[$it]}"
        # echo /home/lth/lth/resnet50_expts/reinit_best_long.sh "${next}" "$j" "resnet50/prune_global_20/v10/lottery/prune_11259/trial_${j}/iter_${it}" "${DENSITY}"
        echo /home/lth/lth/resnet50_expts/reinit_oneshot_long.sh "${next}" "$j" "resnet50/prune_global_20/v10/lottery/prune_11259/trial_${j}/iter_0" "${DENSITY}"
    done
done
