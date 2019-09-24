set -ex

NETWORK="resnet50"
BASE_PRUNE_METHOD="prune_global_20"
VERSION="v10"
FS_PREFIX="gs://REDACTED"
PRUNE_ITERATIONS=10


function is_first_attempt() {
    if gsutil ls "${FS_PREFIX}/execution_data/${CURR_NAME}" >/dev/null 2>/dev/null; then
        return 1
    else
        return 0
    fi
}

function set_prune_method() {
    RATIO="$(python -c "print('{:.3f}'.format(100*.8 ** $1))")"
    PRUNE_METHOD="prune_all_to_global_${RATIO}"
}

DEBUG=""
# DEBUG="yes"

function run_or_debug() {
    if [[ "${DEBUG}" == "yes" ]]; then
        : "${@}"
    else
        "${@}"
        return "$?"
    fi
}
