#!/usr/bin/env bash

gsutil cat "gs://REDACTED/queue_logs/$1.${2-stderr}"'*' | less +G
