#!/usr/bin/env bash

set -eu

DENSITIES=(80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74 8.59 6.87 5.5 4.4 3.52 2.81 2.25 1.8 1.44 1.15 0.92 0.74 0.59 0.47 0.38)
BEST_PRUNE=(80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74 8.59 6.87 5.5 4.4 3.52 2.81 2.25 1.8 1.44 1.15 0.92 0.74 0.59 0.47 0.38)
VERSION="v6"
for VGG_SIZE in "16_nofc" "19_nofc"; do
    for j in 1 2 3; do
        for it in `seq 0 24`; do
            next="$(expr ${it} + 1)"
            DENSITY="${DENSITIES[$it]}"
            PRIDX="${BEST_PRUNE[$it]}"
            # echo /home/lth/lth/resnet20_expts/reinit_best_long.sh "${next}" "$j" "resnet20/prune_global_20/v3/lottery/prune_${PRIDX}/trial_${j}/iter_${it}" "${DENSITY}"
            echo vgg_expts_ibm/reinit_oneshot_long.sh "${VGG_SIZE}" "${VERSION}" 1 "${next}" "$j" "vgg_${VGG_SIZE}/prune_global_20/v6/base/trial_${j}" "${DENSITY}"
        done
    done
done
