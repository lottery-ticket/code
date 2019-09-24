#!/usr/bin/env bash

LTH_DIR="$(dirname $0)/.."

source "$(dirname $0)/common.sh"

if [ "$#" -lt 1 ]; then
    echo "Must provide an experiment name and a TPU index!"
    exit 1
fi

EXPERIMENT_NAME="$1"; shift

MODEL_DIR="${FS_PREFIX}/execution_data/${EXPERIMENT_NAME}";
RESULTS_DIR="${FS_PREFIX}/results/${EXPERIMENT_NAME}";

INSTANCE_NAME="$(python3 -c "import requests;print(requests.get('http://metadata.google.internal/computeMetadata/v1/instance/name',headers={'Metadata-Flavor': 'Google'}).text)")"
INSTANCE_ZONE="$(python3 -c "import requests;print(requests.get('http://metadata.google.internal/computeMetadata/v1/instance/zone',headers={'Metadata-Flavor': 'Google'}).text.split('/')[-1])")"

case "${INSTANCE_ZONE}" in
    "us-"*)
        DATA_DIR="gs://REDACTED-imagenet/tensorflow/tensorflow"
        ;;
    "europe-"*)
        DATA_DIR="gs://imagenet-but-its-in-europe-this-time/tensorflow/tensorflow"
        ;;
    *)
        echo "Unsupported instance zone ${INSTANCE_ZONE}"
        exit 1
        ;;
esac

"${LTH_DIR}/reserve_tpu.py" cleanup
TPU_INDEX=$("${LTH_DIR}/reserve_tpu.py" reserve "$$")
TPU_NAME="${INSTANCE_NAME}-${TPU_INDEX}"

cd "${LTH_DIR}/tpu-src/models"
set +e

# run_or_debug "${LTH_DIR}/REDACTED/cloud.py" slack "\`${QUEUE_PID_NAME:-unknown}\` running: \`${MODEL_DIR}\`"

run_or_debug "${LTH_DIR}/tpu_runner.py" "${TPU_NAME}" python "official/resnet/resnet_main.py" \
       --hparams_file official/resnet/configs/cloud/v3-8.yaml \
       --data_dir "${DATA_DIR}" \
       --resnet_depth 50 \
       --model_dir "${MODEL_DIR}" \
       --lottery_results_dir "${RESULTS_DIR}" \
       --tpu "${TPU_NAME}" \
       "${@}"
status=$?
"${LTH_DIR}/reserve_tpu.py" release "${TPU_INDEX}"
exit $status
