#!/usr/bin/env python3

import argparse
import os
import sys
import fileinput
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+')
parser.add_argument('--collect', action='store_true', default=False)
parser.add_argument('--cache', action='store_true', default=False)
parser.add_argument('--ignore-cache', action='store_true', default=False)
parser.add_argument('--collect-then-run', action='store_true', default=False)

args = parser.parse_args()

def _run(in_, out_):
    n = in_.get()
    try:
        res = utils.iter_datum_of_iter_dir(*n)
    except:
        res = None
    out_.put(res)
    in_.task_done()

def mapit(its, n=None):
    if n is None:
        n = multiprocessing.cpu_count()

    in_ = multiprocessing.JoinableQueue()
    out_ = multiprocessing.Queue()

    procs = []
    def _start():
        proc = multiprocessing.Process(target=_run, args=(in_, out_))
        proc.start()
        procs.append(proc)

    for _ in range(n):
        _start()

    for i in its:
        in_.put(i)

    res = []
    while len(res) < len(its):
        print('{}/{}'.format(len(res), len(its)), end='\r')
        r = out_.get()
        res.append(r)
        _start()

    print('')

    in_.join()
    for p in procs:
        p.terminate()

    return [r for r in res if r is not None]

import utils
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

dirs = [q.strip() for d in args.dirs for q in (fileinput.FileInput(d) if os.path.exists(d) else [d])]
print(dirs, file=sys.stderr)

if args.collect_then_run:
    p = multiprocessing.Pool()
    trials = utils.collect_trials(dirs)
    trial_iters = p.map(tf.gfile.ListDirectory, trials)
    iters = [os.path.join(t, i).rstrip('/') for (t, iters) in zip(trials, trial_iters) for i in iters]
    iters = mapit([((i, args.cache), True, args.ignore_cache) for i in iters])
    print('\n'.join(map('{0[1]} {0[0].density_ratio} {0[0].test_acc}'.format, [(res, expt) for (res, expt) in zip(iters, dirs) if res is not None])))
elif args.collect:
    print('\n'.join(utils.collect_trials(dirs)))
else:
    iters = mapit([((d, args.cache), True, args.ignore_cache) for d in dirs])
    print('\n'.join(map('{0[1]} {0[0].density_ratio} {0[0].test_acc}'.format, [(res, expt) for (res, expt) in zip(iters, dirs) if res is not None])))
