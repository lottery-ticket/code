#!/usr/bin/env bash

./iters.py \
    gs://REDACTED/execution_data/resnet20/prune_global_20/v3/finetune \
    gs://REDACTED/execution_data/resnet20/prune_global_20/v3/lottery \
    gs://REDACTED/execution_data/resnet20/prune_global_20/v4 \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/finetune \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/lottery \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/oneshot_finetune \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/oneshot_lottery \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v11/oneshot_real_reinit \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/reinit_best_long \
    gs://REDACTED/execution_data/resnet50/prune_global_20/v10/reinit_oneshot_long

export AWS_LOG_LEVEL=3
export AWS_REGION=us-standard
export S3_ENDPOINT=s3.us.cloud-object-storage.appdomain.cloud
export AWS_ACCESS_KEY_ID=dcd241d33c7142d6b8a80251e05375fc
export AWS_SECRET_ACCESS_KEY=d30074e169ee8aa885dba41819c247118e163878f47abc32
export S3_USE_HTTPS=1
export S3_VERIFY_SSL=0

./iters.py \
    s3://REDACTED-data/execution_data/vgg_19_nofc/prune_global_20/v6/base \
    s3://REDACTED-data/execution_data/vgg_19_nofc/prune_global_20/v6/reinit_oneshot_long \
    s3://REDACTED-data/execution_data/vgg_19_nofc/prune_global_20/v8 \
    s3://REDACTED-data/execution_data/vgg_16_nofc/prune_global_20/v6/base \
    s3://REDACTED-data/execution_data/vgg_16_nofc/prune_global_20/v6/reinit_oneshot_long \
    s3://REDACTED-data/execution_data/vgg_16_nofc/prune_global_20/v8


export S3_ENDPOINT=s3.eu-gb.cloud-object-storage.appdomain.cloud

./iters.py \
    s3://REDACTED-data-eu-gb/execution_data/vgg_19_nofc/prune_global_20/v6/base \
    s3://REDACTED-data-eu-gb/execution_data/vgg_19_nofc/prune_global_20/v6/reinit_oneshot_long \
    s3://REDACTED-data-eu-gb/execution_data/vgg_19_nofc/prune_global_20/v8 \
    s3://REDACTED-data-eu-gb/execution_data/vgg_16_nofc/prune_global_20/v6/base \
    s3://REDACTED-data-eu-gb/execution_data/vgg_16_nofc/prune_global_20/v6/reinit_oneshot_long \
    s3://REDACTED-data-eu-gb/execution_data/vgg_16_nofc/prune_global_20/v8
