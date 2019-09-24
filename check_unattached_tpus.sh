#!/usr/bin/env bash

DIRNAME="$(dirname $0)"
TPUSTR="$(gcloud compute tpus list --zone us-central1-a | grep '^expt' | awk '{print $1}')"

IFS=$'\n' read -rd '' -a TPUS <<<"$TPUSTR"

good=()

containsElement () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}


for TPU in "${TPUS[@]}"; do
    INST="$(echo ${TPU} | sed 's/-.*//g')"
    if containsElement "${INST}" "${good[@]}"; then
        continue
    fi

    if "${DIRNAME}/REDACTED/cloud.py" connect "${INST}" --com false 1>/dev/null 2>/dev/null; then
        echo "${TPU}"
    else
        good+=("${INST}")
    fi
done
