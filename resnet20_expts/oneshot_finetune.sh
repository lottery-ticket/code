#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 4 ]; then
    echo "Must provide a ft epochs, a trial index, a prev name (e.g. 'vgg/prune/lottery/prune_4701/trial_1'), and a density rate (e.g. 80.4)!"
    exit 1
fi

FINETUNE_EPOCHS="$1"
TRIAL_INDEX="$2"
PREV_NAME="$3"
PRUNE_RATE="$4"

TRAIN_STEPS="$(expr \( ${FINETUNE_EPOCHS} \* \( 50000 / 128 \) \) + 71108 || :)"

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/oneshot_finetune/finetune_${FINETUNE_EPOCHS}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

if is_first_attempt; then
    run_or_debug gsutil -m cp \
                 "${FS_PREFIX}/execution_data/${PREV_NAME}/graph.pbtxt" \
                 "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"'*' \
                 "${FS_PREFIX}/execution_data/${CURR_NAME}/"

    TMPFILE="$(mktemp)"
    echo 'model_checkpoint_path: "'"checkpoint_iter_final"'"' > "${TMPFILE}"
    run_or_debug gsutil cp "${TMPFILE}" "${FS_PREFIX}/execution_data/${CURR_NAME}/checkpoint"
    rm "${TMPFILE}"
fi


run_or_debug "$(dirname $0)/run_base.sh" "${CURR_NAME}" \
    --lottery_pruning_method "${PRUNE_METHOD}" \
    --lottery_reset_to "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" \
    --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final" \
    --lottery_force_learning_rate "0.001" \
    --max_train_steps "${TRAIN_STEPS}"
