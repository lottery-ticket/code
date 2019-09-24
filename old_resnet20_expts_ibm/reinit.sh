#!/usr/bin/env bash

source "$(dirname $0)/common.sh"

if [ "$#" -ne 3 ]; then
    echo "Must provide a pruning index, a reinit index, and a trial index!"
    exit 1
fi

PRUNE_INDEX="$1"
REINIT_INDEX="$2"
TRIAL_INDEX="$3"
TRAIN_EPOCHS="$(expr \( 71108 - ${PRUNE_INDEX} \) / \( 50000 / 128 \) || :)"
FULL_TRAIN_EPOCHS="$(expr 182 + \( ${REINIT_INDEX} \* ${TRAIN_EPOCHS} \) || :)"

BASENAME="${NETWORK}/${PRUNE_METHOD}/${VERSION}/reinit/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}"
PREVNAME="${NETWORK}/${PRUNE_METHOD}/${VERSION}/lottery/prune_${PRUNE_INDEX}/trial_${TRIAL_INDEX}"
PREVITER="$(expr ${REINIT_INDEX} - 1 || :)"

"$(dirname $0)/run_base.sh" \
    "${BASENAME}/iter_${REINIT_INDEX}" \
    --lottery_prune_at "${FS_PREFIX}/results/${PREVNAME}/iter_${PREVITER}/checkpoint_iter_final" \
    --lottery_pruning_method "${PRUNE_METHOD}" \
    --lottery_reset_to "reinitialize" \
    --train_epochs "${FULL_TRAIN_EPOCHS}"
