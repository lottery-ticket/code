#!/usr/bin/env bash

set -ex

rm -rf figures
./unified_plot.py
rm figures.zip || :
zip -r figures.zip figures
gsutil cp figures.zip gs://REDACTED-public/tinylth-paper/figures.zip
gsutil acl ch -u AllUsers:R gs://REDACTED-public/tinylth-paper/figures.zip
