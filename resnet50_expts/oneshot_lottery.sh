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

CURR_NAME="${NETWORK}/${BASE_PRUNE_METHOD}/${VERSION}/oneshot_lottery/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}/prune_${PRUNE_RATE}"

PRUNE_METHOD="prune_all_to_global_${PRUNE_RATE}"

if is_first_attempt; then
    run_or_debug gsutil -m cp "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}"'*' "${FS_PREFIX}/results/${CURR_NAME}/"
    run_or_debug gsutil -m cp "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}"'*' "${FS_PREFIX}/execution_data/${CURR_NAME}/"
    run_or_debug gsutil -m cp -r "${FS_PREFIX}/execution_data/${PREV_NAME}/graph.pbtxt" "${FS_PREFIX}/execution_data/${CURR_NAME}/"

    TEMP_CHECKPOINT="$(mktemp)"
    echo 'model_checkpoint_path: "'"checkpoint_iter_${PRUNE_INDEX}"'"' > "${TEMP_CHECKPOINT}"
    run_or_debug gsutil cp "${TEMP_CHECKPOINT}" "${FS_PREFIX}/execution_data/${CURR_NAME}/checkpoint"
    rm "${TEMP_CHECKPOINT}"
fi


run_or_debug "$(dirname $0)/run_base.sh" "${CURR_NAME}" \
    --train_steps 112590 \
    --lottery_checkpoint_iters "${PRUNE_INDEX}" \
    --lottery_pruning_method "${PRUNE_METHOD}" \
    --lottery_reset_to "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_${PRUNE_INDEX}" \
    --lottery_prune_at "${FS_PREFIX}/results/${PREV_NAME}/checkpoint_iter_final"
