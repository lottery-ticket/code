#!/usr/bin/env bash

function enqueue() {
    echo "${@}"
}

for j in 1 2 3; do
    for i in 0 11259 22518 33777 45036 56295 67554 78813 90072 101331 112590; do
        enqueue /home/lth/lth/resnet50_expts/lottery.sh "$i" "$j"
        enqueue /home/lth/lth/resnet50_expts/finetune.sh "$(expr \( 112590 - $i \) / 1251 )" "$j"
    done
    for i in 0 11259 22518 33777 45036 56295 67554 78813 90072 101331; do
        enqueue /home/lth/lth/resnet50_expts/finetune.sh "$(expr \( \( 2 \* 112590 \) - $i \) / 1251 )" "$j"
    done

    for DENSITY in 80.0 64.0 51.2 40.96 32.77 26.21 20.97 16.78 13.42 10.74; do
        for i in 0 11259 22518 33777 45036 56295 67554 78813 90072 101331 112590; do
            TRIAL_DIR="resnet50/prune_global_20/v10/finetune/finetune_9/trial_${j}/iter_0"
            train_iters="$(expr 112590 - $i)"
            enqueue /home/lth/lth/resnet50_expts/oneshot_lottery.sh "$i" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet50_expts/oneshot_finetune.sh "$(expr ${train_iters} / 1251 )" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet50_expts/oneshot_reinit.sh "${train_iters}" "$j" "${TRIAL_DIR}" "${DENSITY}"
        done
        for i in 0 11259 22518 33777 45036 56295 67554 78813 90072 101331; do
            train_iters="$(expr \( 112590 \* 2 \) - $i)"
            enqueue /home/lth/lth/resnet50_expts/oneshot_finetune.sh "$(expr ${train_iters} / 1251 )" "$j" "${TRIAL_DIR}" "${DENSITY}"
            enqueue /home/lth/lth/resnet50_expts/oneshot_reinit.sh "${train_iters}" "$j" "${TRIAL_DIR}" "${DENSITY}"
        done
    done
done
