#!/usr/bin/env bash

gcloud --project carbingroup pubsub topics publish REDACTED --message "$*"
