#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 4 ]; then
    echo "Must provide a pruning index, a trial index, and a prev name (e.g. 'vgg/prune/lottery/prune_4701/trial_1')!"
    exit 1
fi

PRUNE_INDEX="$1"
TRIAL_INDEX="$2"
PREV_NAME="$3"
PRUNE_RATE="$4"

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/oneshot_lottery/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"
PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

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


run_base "${CURR_NAME}" \
         --max_train_steps "71108"\
         --lottery_pruning_method "${PRUNE_METHOD}" \
         --lottery_reset_to "${FS_PREFIX}/results/${PREV_NAME}/iter_0/checkpoint_iter_${PRUNE_INDEX}" \
         --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/iter_0/checkpoint_iter_final"
