#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 2 ]; then
    echo "Must provide a pruning index and a trial index!"
    exit 1
fi

PRUNE_INDEX="$1"
TRIAL_INDEX="$2"
TRAIN_EPOCHS="$(expr \( 71108 - $PRUNE_INDEX \) / \( 50000 / 128 \) || :)"
RESET_PRUNE_INDEX_APPROX="$(expr ${PRUNE_INDEX} - \( 40 \* 50000 / 128 \) || :)"

for i in 0 14 404 795 1576 4701 8608 12514 16420 20326 24233 28139 32045 35951 39858 43764 47670 51576 55483 59389 63295 67201 69545 70326 70717 71108; do
    diff="$(expr ${RESET_PRUNE_INDEX_APPROX} - ${i} || :)"
    if [ "${diff#-}" -lt 5 ]; then
        RESET_PRUNE_INDEX=$i
        break
    fi
done

if [ -z "${RESET_PRUNE_INDEX}" ]; then
    echo "Prune index ${PRUNE_INDEX} cannot be shortcut by 40"
    exit 1
fi

BASENAME="${NETWORK}/${PRUNE_METHOD}/${VERSION}/lottery_early_40/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}"

function name_of_seq() {
    echo "${BASENAME}/iter_$1"
}

for idx in `seq 0 ${PRUNE_ITERATIONS}`; do
    PREV_NAME="$(name_of_seq $(expr $idx - 1))"

    if [ "$idx" -eq 0 ]; then
        extra_params=()
    else
        extra_params=("--lottery_reset_to" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${RESET_PRUNE_INDEX}" "--lottery_prune_at" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" "--train_epochs" "$TRAIN_EPOCHS")
    fi

    "$(dirname $0)/run_base.sh" "$(name_of_seq $idx)" --lottery_pruning_method "${PRUNE_METHOD}" "${extra_params[@]}"
done
