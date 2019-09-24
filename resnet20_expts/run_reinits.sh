#!/usr/bin/env bash

DENSITIES=(80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74)
BEST_PRUNE=(4701 4701 4701 28139 4701 12514 12514 12514 12514 12514)
for j in 1 2 3; do
    for it in `seq 0 9`; do
        next="$(expr ${it} + 1)"
        DENSITY="${DENSITIES[$it]}"
        PRIDX="${BEST_PRUNE[$it]}"
        # echo /home/lth/lth/resnet20_expts/reinit_best_long.sh "${next}" "$j" "resnet20/prune_global_20/v3/lottery/prune_${PRIDX}/trial_${j}/iter_${it}" "${DENSITY}"
        echo /home/lth/lth/resnet20_expts/reinit_oneshot_long.sh "${next}" "$j" "resnet20/prune_global_20/v3/lottery/prune_4701/trial_${j}/iter_0" "${DENSITY}"
    done
done
