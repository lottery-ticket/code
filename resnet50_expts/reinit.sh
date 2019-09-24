#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 5 ]; then
    echo "Must provide a prune index, a trial index, a prev name (e.g. 'vgg/prune/lottery/prune_4701/trial_1'), a density rate (e.g. 80.4), and the number of steps to train!"
    exit 1
fi

PRUNE_INDEX="$1"
TRIAL_INDEX="$2"
PREV_NAME="$3"
PRUNE_RATE="$4"
TRAIN_STEPS="$5"

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/reinit/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

run_or_debug "$(dirname $0)/run_base.sh" "${CURR_NAME}" \
    --train_steps "${TRAIN_STEPS}"
    --lottery_checkpoint_iters "${PRUNE_INDEX}" \
    --lottery_pruning_method "${PRUNE_METHOD}" \
    --lottery_reset_to "${FS_PREFIX}/results/${PREV_NAME}/iter_0/checkpoint_iter_${PRUNE_INDEX}" \
    --lottery_prune_at "reinitialize"
