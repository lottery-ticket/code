#!/usr/bin/env bash

set -u

VERSION="v6"

for VGG_SIZE in 16_nofc 19_nofc; do
    for j in 1 2 3; do
        for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108; do
            epoch="$(expr \( 71108 - $i \) / \( 50000 / 128 \))"
            echo "vgg_expts_ibm/lottery.sh" "${VGG_SIZE} ${VERSION} 30 ${i} ${j}"
            echo "vgg_expts_ibm/finetune.sh" "${VGG_SIZE} ${VERSION} 30 ${epoch} ${j}"
        done
        for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201; do
            epoch="$(expr \( \( 2 \* 71108 \) - $i \)  / \( 50000 / 128 \))"
            echo "vgg_expts_ibm/finetune.sh" "${VGG_SIZE} ${VERSION} 30 ${epoch} ${j}"
        done

        PREV_NAME="vgg_${VGG_SIZE}/prune_global_20/v6/base/trial_${j}"

        for prune in 80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74 8.59 6.87 5.5 4.4 3.52 2.81 2.25 1.8 1.44 1.15 0.92 0.74 0.59 0.47 0.38; do
            for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108; do
                epoch="$(expr \( 71108 - $i \)  / \( 50000 / 128 \))"
                echo "vgg_expts_ibm/oneshot_lottery.sh" "${VGG_SIZE} ${VERSION} 1 ${i} ${j} ${PREV_NAME} ${prune}"
                echo "vgg_expts_ibm/oneshot_finetune.sh" "${VGG_SIZE} ${VERSION} 1 ${epoch} ${j} ${PREV_NAME} ${prune}"
                echo "vgg_expts_ibm/oneshot_reinit.sh" "${VGG_SIZE} ${VERSION} 1 ${epoch} ${j} ${PREV_NAME} ${prune}"
            done
            for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201; do
                epoch="$(expr \( \( 2 \* 71108 \) - $i \)  / \( 50000 / 128 \))"
                echo "vgg_expts_ibm/oneshot_finetune.sh" "${VGG_SIZE} ${VERSION} 1 ${epoch} ${j} ${PREV_NAME} ${prune}"
                echo "vgg_expts_ibm/oneshot_reinit.sh" "${VGG_SIZE} ${VERSION} 1 ${epoch} ${j} ${PREV_NAME} ${prune}"
            done
        done
    done
done
