#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 4 ]; then
    echo "Must provide an iteration, a trial index, a prev name (e.g. 'vgg/prune/lottery/prune_4701/trial_1'), and a density rate (e.g. 80.4)!"
    exit 1
fi

ITERATION="$1"
TRIAL_INDEX="$2"
PREV_NAME="$3"
PRUNE_RATE="$4"

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/reinit_best_long/retrain_${ITERATION}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

for i in 0 9 18 27 36 45 54 63 72 81 90; do
    TRAIN_STEPS="$(expr 112590 + \( ${i} \* 1251 \* ${ITERATION} \))"
    run_or_debug "$(dirname $0)/run_base.sh" "${CURR_NAME}" \
                 --train_steps "${TRAIN_STEPS}" \
                 --lottery_pruning_method "${PRUNE_METHOD}" \
                 --lottery_reset_to "reinitialize" \
                 --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
done
