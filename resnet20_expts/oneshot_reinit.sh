#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 4 ]; then
    echo "Must provide a prune index, a trial index, a prev name (e.g. 'vgg/prune/lottery/prune_4701/trial_1'), and a density rate (e.g. 80.4)!"
    exit 1
fi

PRUNE_INDEX="$1"
TRIAL_INDEX="$2"
PREV_NAME="$3"
PRUNE_RATE="$4"
FULL_TRAIN_STEPS="$(expr 71108 + ${PRUNE_INDEX})"

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/oneshot_real_reinit/retrain_${PRUNE_INDEX}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

run_or_debug "$(dirname $0)/run_base.sh" "${CURR_NAME}" \
    --max_train_steps "${FULL_TRAIN_STEPS}" \
    --lottery_pruning_method "${PRUNE_METHOD}" \
    --lottery_reset_to "reinitialize" \
    --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
