#!/usr/bin/env python3

import argparse
import os
import multiprocessing

def meets_thresh(curr_min, new_val, threshold):
    if threshold.endswith('%'):
        improvement = 1 - new_val / curr_min
        return improvement >= float(threshold[:-1]) / 100
    else:
        return curr_min - new_val >= float(threshold)

def get_losses(events_file):
    import tensorflow as tf
    res = []
    for e in tf.train.summary_iterator(events_file):
        for v in e.summary.value:
            if 'loss' in v.tag:
                res.append((e.step, v.simple_value))
    return res

def listdir(directory):
    import tensorflow as tf
    return list(tf.gfile.ListDirectory(directory))

def check_dir(directory, iterations_without_improvement, improvement_threshold):

    res = []

    p = multiprocessing.Pool(1)
    event_file_candidates = p.apply(listdir, [directory])
    p.close()

    for events_file in event_file_candidates:
        if not events_file.startswith('events.out'):
            continue
        p = multiprocessing.Pool(1)
        res.extend(p.apply(get_losses, [os.path.join(directory, events_file)]))
        p.close()

    res = sorted(res)

    if len(res) == 0:
        return False

    thresh_iter, thresh_min = res[0]
    print(res[0])

    for it, val in res:
        if meets_thresh(thresh_min, val, improvement_threshold):
            thresh_iter, thresh_min = it, val
            print((it, val))

    print(res[-1][0])
    return res[-1][0] - thresh_iter > iterations_without_improvement

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('iterations_without_improvement', type=int)
    parser.add_argument('improvement_threshold')

    args = parser.parse_args()
    print(check_dir(args.directory, args.iterations_without_improvement, args.improvement_threshold))

if __name__ == '__main__':
    main()
