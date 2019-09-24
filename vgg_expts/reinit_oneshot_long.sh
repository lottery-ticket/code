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

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/reinit_oneshot_long/retrain_${ITERATION}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

for i in 0 10 30 50 70 90 110 130 150 170 182; do
    TRAIN_STEPS="$(expr 71108 + \( ${i} \* ${ITERATION} \* \( 50000 / 128 \) \))"
    run_base "${CURR_NAME}" \
             --max_train_steps "${TRAIN_STEPS}" \
             --lottery_pruning_method "${PRUNE_METHOD}" \
             --lottery_reset_to "reinitialize" \
             --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/iter_0/checkpoint_iter_final"
done
