#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 2 ]; then
    echo "Must provide a pruning index and a trial index!"
    exit 1
fi

PRUNE_INDEX="$1"
if [ "${PRUNE_INDEX}" -eq 71108 ]; then
    exit 0
fi

TRIAL_INDEX="$2"

BASENAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/lottery/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}"

function name_of_seq() {
    echo "${BASENAME}/iter_$1"
}

for idx in `seq ${PRUNE_START_ITER} ${PRUNE_ITERATIONS}`; do
    PREV_NAME="$(name_of_seq $(expr $idx - 1))"
    CURR_NAME="$(name_of_seq $idx)"

    set_prune_method "${idx}"

    if [ "$idx" -eq 0 ]; then
        extra_params=()
    else
        if is_first_attempt; then
            mkdir -p "${FS_PREFIX}/results/${CURR_NAME}/" "${FS_PREFIX}/execution_data/${CURR_NAME}/"

            cp "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}"* "${FS_PREFIX}/results/${CURR_NAME}/"
            cp "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}"* "${FS_PREFIX}/execution_data/${CURR_NAME}/"
            cp "${FS_PREFIX}/execution_data/${PREV_NAME}/graph.pbtxt" "${FS_PREFIX}/execution_data/${CURR_NAME}/"

            TEMP_CHECKPOINT="$(mktemp)"
            echo 'model_checkpoint_path: "'"checkpoint_iter_${PRUNE_INDEX}"'"' > "${TEMP_CHECKPOINT}"
            cp "${TEMP_CHECKPOINT}" "${FS_PREFIX}/execution_data/${CURR_NAME}/checkpoint"
            rm "${TEMP_CHECKPOINT}"
        fi

        extra_params=(
            "--lottery_reset_to" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}"
            "--lottery_prune_at" "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
        )
    fi

    run_base "$(name_of_seq $idx)" \
             --lottery_pruning_method "${PRUNE_METHOD}" \
             --max_train_steps 71108 \
             "${extra_params[@]}"
done
