#!/usr/bin/env bash

set -eux

for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108; do
    for j in 1 2 3; do
        ./run_ibm_job.sh resnet20_expts_ibm/lottery.sh "$i" "$j"
        ./run_ibm_job.sh resnet20_expts_ibm/abbreviated.sh "$i" "$j"
        ./run_ibm_job.sh resnet20_expts_ibm/finetune.sh "$(expr \( 71108 - $i \) / \( 50000 / 128 \) )" "$j"
    done
done

# q=0

# for j in 1 2 3; do
#     for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108; do
#         for k in `seq 3 7`; do
#             if [ ! $q -lt 33 ]; then
#                 ./run_ibm_job.sh resnet20_expts_ibm/reinit.sh "$i" "$k" "$j"
#             fi
#             q=`expr $q + 1`
#         done
#     done
# done
