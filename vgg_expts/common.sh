set -ex

if [ "$#" -lt 3 ]; then
    echo "Must provide a VGG_SIZE, a VERSION, and PRUNE_ITERATIONS!"
    exit 1
fi

VGG_SIZE="$1"; shift
VERSION="$1"; shift
PRUNE_ITERATIONS="$1"; shift

BASE_PRUNE_METHOD="prune_global_20"

case "${VGG_SIZE}" in
    *_fc)
        VGG_SIZE="${VGG_SIZE: : -3}"
        ;;
esac


# case "${VGG_SIZE}" in
#     11 | 16 | 19 | 11_nofc | 16_nofc | 19_nofc) ;;
#     *) echo "Unknown VGG_SIZE \"${VGG_SIZE}\""
#        exit 1
#        ;;
# esac

case "${VERSION}" in
    v[0-9]* ) ;;
    *) echo "Illegal VERSION \"${VERSION}\""
       exit 1
       ;;
esac

NETWORK="vgg_${VGG_SIZE}"
FS_PREFIX="${DATA_DIR}"
PRUNE_START_ITER="0"

function run_base() {
    "$(dirname $0)/run_base.sh" \
        "${VGG_SIZE}" "${VERSION}" "${PRUNE_ITERATIONS}" \
        "${@}"
}

function is_first_attempt() {
    if ls "${FS_PREFIX}/execution_data/${CURR_NAME}" >/dev/null 2>/dev/null; then
        return 1
    else
        return 0
    fi
}

function set_prune_method() {
    RATIO="$(python -c "print('{:.3f}'.format(100*.8 ** $1))")"
    PRUNE_METHOD="prune_all_to_global_${RATIO}"
}
