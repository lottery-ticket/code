#!/usr/bin/env bash

set -u

TARGET="$1"

DEST_REGION="$(cat $1 | jq -r '.region')"
DEST_INSTANCE="$(cat $1 | jq -r '.instance.instance_id')"

bx target -r "${DEST_REGION}"
bx ml set instance "${DEST_INSTANCE}"
