#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 2 ]; then
    echo "Must provide a number of finetuning epochs and a trial index!"
    exit 1
fi

FINETUNE_EPOCHS="$1"
TRIAL_INDEX="$2"
TRAIN_STEPS="$(expr ${FINETUNE_EPOCHS} \* 1251 || :)"

BASENAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/finetune/finetune_${FINETUNE_EPOCHS}/trial_${TRIAL_INDEX}"

function name_of_seq() {
    echo "${BASENAME}/iter_$1"
}

if [ "${FINETUNE_EPOCHS}" -eq 0 ]; then
    exit 0
fi

TOTAL_TRAIN_STEPS="112590"

for idx in `seq 0 ${PRUNE_ITERATIONS}`; do
    PREV_NAME="$(name_of_seq $(expr $idx - 1))"
    CURR_NAME="$(name_of_seq $idx)"

    set_prune_method "${idx}"

    if [ "$idx" -eq 0 ]; then
        extra_params=()
    else
        if is_first_attempt; then
            run_or_debug gsutil -m cp "${FS_PREFIX}/execution_data/${PREV_NAME}/graph.pbtxt" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"'*' \
                   "${FS_PREFIX}/execution_data/${CURR_NAME}/"

            TEMP_CHECKPOINT="$(mktemp)"
            echo 'model_checkpoint_path: "'"checkpoint_iter_final"'"' > "${TEMP_CHECKPOINT}"
            run_or_debug gsutil cp "${TEMP_CHECKPOINT}" "${FS_PREFIX}/execution_data/${CURR_NAME}/checkpoint"
            rm "${TEMP_CHECKPOINT}"
        fi
        extra_params=(
            "--lottery_reset_to" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
            "--lottery_prune_at" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
            "--lottery_force_learning_rate" "0.0004"
        )
    fi

    run_or_debug "$(dirname $0)/run_base.sh" "$(name_of_seq $idx)" --lottery_pruning_method "${PRUNE_METHOD}" --train_steps "${TOTAL_TRAIN_STEPS}" "${extra_params[@]}"

    TOTAL_TRAIN_STEPS="$(expr ${TOTAL_TRAIN_STEPS} + ${TRAIN_STEPS})"
done
