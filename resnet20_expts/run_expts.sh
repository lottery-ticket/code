#!/usr/bin/env bash

function enqueue() {
    echo "${@}"
}

for j in 1 2 3; do
    for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108 ; do
        enqueue /home/lth/lth/resnet20_expts_gcp/lottery.sh "$i" "$j"
        enqueue /home/lth/lth/resnet20_expts_gcp/finetune.sh "$(expr \( 71108 - $i \) / \( 50000 / 128 \) )" "$j"
    done

    for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201; do
        enqueue /home/lth/lth/resnet20_expts_gcp/finetune.sh "$(expr \( \( 2 \* 71108 \) - $i \) / \( 50000 / 128 \) )" "$j"
    done

    TRIAL_DIR="resnet20/prune_global_20/v2/finetune/finetune_90/trial_${j}/iter_0"
    for DENSITY in 80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74; do
        for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201 71108; do
            train_iters="$(expr 71108 - $i)"
            enqueue /home/lth/lth/resnet20_expts_gcp/oneshot_lottery.sh "$i" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet20_expts_gcp/oneshot_finetune.sh "$(expr ${train_iters} / \( 50000 / 128 \) )" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet20_expts_gcp/oneshot_reinit.sh "${train_iters}" "$j" "${TRIAL_DIR}" "${DENSITY}"
        done
        for i in 0 4701 12514 20326 28139 35951 43764 51576 59389 67201; do
            train_iters="$(expr \( 2 \* 71108 \) - $i)"
            enqueue /home/lth/lth/resnet20_expts_gcp/oneshot_finetune.sh "$(expr ${train_iters} / \( 50000 / 128 \) )" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet20_expts_gcp/oneshot_reinit.sh "${train_iters}" "$j" "${TRIAL_DIR}" "${DENSITY}"
        done
    done

done
