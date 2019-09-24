#!/usr/bin/env bash

tf=$(mktemp)

./load_ibm_data.py > $tf

cat $tf | wc -l

cat $tf | xargs -s65536 ./read_from_s3.py --region us-standard --endpoint s3.us.cloud-object-storage.appdomain.cloud --access-key-id dcd241d33c7142d6b8a80251e05375fc --secret-access-key d30074e169ee8aa885dba41819c247118e163878f47abc32 --cache

cat $tf | xargs -s65536 ./read_from_s3.py --region us-standard --endpoint s3.eu-gb.cloud-object-storage.appdomain.cloud --access-key-id dcd241d33c7142d6b8a80251e05375fc --secret-access-key d30074e169ee8aa885dba41819c247118e163878f47abc32 --cache

rm $tf
