#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 2 ]; then
    echo "Must provide a pruning index and a trial index!"
    exit 1
fi

PRUNE_INDEX="$1"
TRIAL_INDEX="$2"
TRAIN_EPOCHS="$(expr \( 71108 - ${PRUNE_INDEX} \) / \( 50000 / 128 \) || :)"

BASENAME="${NETWORK}/${PRUNE_METHOD}/${VERSION}/lottery/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}"

function name_of_seq() {
    echo "${BASENAME}/iter_$1"
}

for idx in `seq 0 ${PRUNE_ITERATIONS}`; do
    PREV_NAME="$(name_of_seq $(expr $idx - 1))"

    if [ "$idx" -eq 0 ]; then
        extra_params=()
    else
        extra_params=("--lottery_reset_to" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}" "--lottery_prune_at" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" "--train_epochs" "$TRAIN_EPOCHS")
    fi

    "$(dirname $0)/run_base.sh" "$(name_of_seq $idx)" --lottery_pruning_method "${PRUNE_METHOD}" "${extra_params[@]}"
done
