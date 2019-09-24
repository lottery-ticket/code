#!/usr/bin/env zsh

set -ex

COUNT=$1; shift
if [ -z "$1" ]; then
    NTPU=0
elif [ "$1" = "--tpu" ]; then
    shift
    if [ -z "$1" ]; then
        NTPU=1
    else
        NTPU="$1"
    fi
else
    echo "Unknown argument $1"
    exit 1
fi

ZONE="$(python -c 'import json; print(json.load(open("cloud_config.json"))["gcp_config"]["zone"])')"

running_jobs=()

for i in $(seq 1 $COUNT); do

    (
        if [ $i -ne $COUNT ]; then
            exec 1>/dev/null
            exec 2>/dev/null
        fi
        uuid="$(uuidgen | cut -c1-16 | tr '[:upper:]' '[:lower:]')"
        NAME="queue${${uuid}//-/}"
        if [ "${NTPU}" -gt 0 ]; then
            ACCELERATOR_ARGS=("-a" "tpu-pre-v3-8:${NTPU}")
            N_QUEUES="${NTPU}"
        else
            ACCELERATOR_ARGS=("-a" "p100:1")
            N_QUEUES="1"
        fi
        REDACTED/cloud.py start -n "${NAME}" -t n1-standard-2 "${ACCELERATOR_ARGS[@]}"
        REDACTED/cloud.py connect "${NAME}" --tty --root --com '/home/lth/lth/byteit.py /opt/conda/envs/lth/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so:0x04258794 0x603d0800'
        REDACTED/cloud.py connect "${NAME}" --tty --root --com '/home/lth/lth/byteit.py /opt/conda/envs/lth/lib/python3.6/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so:0x042587e2 0x603d0800'
        REDACTED/cloud.py connect "${NAME}" --tty --root --com 'patch /opt/conda/envs/lth/lib/python3.6/site-packages/tensorflow/contrib/tpu/python/tpu/tpu_system_metadata.py < <(cat <<EOF
32c32
< _RETRY_TIMES = 120
---
> _RETRY_TIMES = 15
EOF
)'
        for i in `seq 1 ${N_QUEUES}`; do
            REDACTED/cloud.py connect "${NAME}" --bg --com "nohup /home/lth/lth/queue_runner.py" --tty
        done
    ) &
    running_jobs+=("$!")
done

for job in "$running_jobs[@]"; do
    wait $job
done
