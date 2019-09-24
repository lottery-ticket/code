#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 2 ]; then
    echo "Must provide a number of finetuning epochs and a trial index!"
    exit 1
fi

FINETUNE_EPOCHS="$1"
TRIAL_INDEX="$2"

BASENAME="${NETWORK}/${PRUNE_METHOD}/${VERSION}/finetune/finetune_${FINETUNE_EPOCHS}/trial_${TRIAL_INDEX}"

function name_of_seq() {
    echo "${BASENAME}/iter_$1"
}

for idx in `seq 0 ${PRUNE_ITERATIONS}`; do
    PREV_NAME="$(name_of_seq $(expr $idx - 1))"

    if [ "$idx" -eq 0 ]; then
        extra_params=()
    else
        extra_params=("--lottery_reset_to" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" "--lottery_prune_at" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" "--train_epochs" "${FINETUNE_EPOCHS}" "--lottery_force_learning_rate" "0.001")
    fi

    "$(dirname $0)/run_base.sh" "$(name_of_seq $idx)" --lottery_pruning_method "${PRUNE_METHOD}" "${extra_params[@]}"
done
