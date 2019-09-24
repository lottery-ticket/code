#!/usr/bin/env bash

ZONE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/zone -HMetadata-Flavor:Google -s | awk -F'/' '{print $4}')
PROJECT=$(curl http://metadata.google.internal/computeMetadata/v1/project/project-id -HMetadata-Flavor:Google -s)
NAME=$(curl http://metadata.google.internal/computeMetadata/v1/instance/name -HMetadata-Flavor:Google -s)

yes | gcloud compute --project $PROJECT instances delete --zone $ZONE $NAME
