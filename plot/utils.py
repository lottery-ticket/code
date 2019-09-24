import collections
import multiprocessing as mp
import os
import re
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pickle
import hashlib
import queue
import sqlite3
import traceback

import gfile

trial_re = re.compile(r'^trial_(?P<trial>\d+)$')
iter_re = re.compile(r'^iter_(?P<iter>\d+)$')

IterDatum = collections.namedtuple('IterDatum', [
    'iter',
    'density_ratio',
    'test_acc',
])
TrialDatum = collections.namedtuple('TrialDatum', [
    'trial',
    'iter_data',
])
Experiment = collections.namedtuple('Experiment', [
    'experiment_dir',
    'trial_data',
])

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_SQLITE_FILE = os.path.join(_DIRNAME, 'iters.db')

if not os.path.exists(_SQLITE_FILE):
    with sqlite3.connect(_SQLITE_FILE) as conn:
        conn.execute("CREATE TABLE iters (path TEXT PRIMARY KEY, density REAL NOT NULL, test_acc REAL NOT NULL)")

def write_iter(directory, density_ratio, test_acc):
    while True:
        try:
            with sqlite3.connect(_SQLITE_FILE) as conn:
                conn.execute('REPLACE INTO iters(path,density,test_acc) VALUES (?,?,?)', (directory.rstrip('/'), density_ratio, test_acc))
            break
        except:
            traceback.print_exc()

def read_iter(directory):
    while True:
        try:
            with sqlite3.connect(_SQLITE_FILE) as conn:
                cur = conn.execute('SELECT path,density,test_acc FROM iters WHERE path=?', (directory.rstrip('/'),))
                return cur.fetchone()
        except:
            traceback.print_exc()

N_WORKERS = 16

def collect_trials_worker(work_queue, results_queue, idx):
    while True:
        directory = work_queue.get()

        if directory is None:
            work_queue.task_done()
            return

        directory = directory.rstrip('/')
        trial_match = trial_re.match(os.path.basename(directory))
        if trial_match:
            results_queue.put(directory)
            work_queue.task_done()
            continue
        elif not gfile.IsDirectory(directory):
            work_queue.task_done()
            continue

        for subdir in map(lambda subdir: os.path.join(directory, subdir), gfile.ListDirectory(directory)):
            work_queue.put(subdir)

        work_queue.task_done()
        continue

def collect_trials(directories):
    collect_trials_q = mp.JoinableQueue()
    collect_trials_results = mp.Queue()

    for directory in directories:
        collect_trials_q.put(directory)

    workers = []
    for idx in range(N_WORKERS):
        w = mp.Process(target=collect_trials_worker, args=(collect_trials_q, collect_trials_results, idx))
        w.start()
        workers.append(w)

    collect_trials_q.join()

    for _ in range(N_WORKERS):
        collect_trials_q.put(None)

    results = []

    while True:
        try:
            results.append(collect_trials_results.get(timeout=1))
        except queue.Empty:
            break

    for w in workers:
        w.join()

    while True:
        try:
            results.append(collect_trials_results.get(timeout=1))
        except queue.Empty:
            break

    return results

def iter_dirs_of_trial_dir(trial_dir):
    return [os.path.join(trial_dir, d) for d in filter(iter_re.match, map(lambda d: d.rstrip('/'), gfile.ListDirectory(trial_dir)))]

def history_of_iter_dir(iter_dir, can_write_cache=False):
    execution_data_iter_dir = os.path.join(iter_dir.replace('results', 'execution_data'), 'eval')
    if not gfile.IsDirectory(execution_data_iter_dir):
        return None

    test_acc = None
    test_iter = None
    for events_file in gfile.ListDirectory(execution_data_iter_dir):
        if not events_file.startswith('events.out'):
            continue
        for e in tf.train.summary_iterator(os.path.join(execution_data_iter_dir, events_file)):
            for v in e.summary.value:
                if v.tag == 'accuracy' or v.tag == 'top_1_accuracy':
                    if test_iter is None or e.step > test_iter:
                        test_iter = e.step
                        test_acc = v.simple_value

    try:
        with gfile.Open(os.path.join(iter_dir, 'density_ratio')) as f:
            density_ratio = float(f.read())
    except Exception as e:
        density_ratio = 1.0

    res = IterDatum(
        iter=os.path.basename(iter_dir),
        density_ratio=density_ratio,
        test_acc=test_acc,
    )

    if can_write_cache and test_acc is not None:
        write_iter(iter_dir, density_ratio, test_acc)
        with gfile.Open(plot_cache, 'w') as f:
            f.write('')
            f.flush()
        with gfile.Open(plot_cache, 'wb') as f:
            pickle.dump(res, f)
    return res


def iter_datum_of_iter_dir(iter_dir_and_can_write_cache, verbose=True, ignore_cache=False):
    iter_dir, can_write_cache = iter_dir_and_can_write_cache

    if not ignore_cache:
        res = read_iter(iter_dir)
        if res:
            return IterDatum(
                iter=os.path.basename(res[0]),
                density_ratio=res[1],
                test_acc=res[2],
            )
        plot_cache = os.path.join(iter_dir, 'plot_cache.pkl')
        if gfile.Exists(plot_cache):
            if verbose:
                print('PLOT CACHE EXISTS: {}'.format(plot_cache))
            with gfile.Open(plot_cache, 'rb') as f:
                try:
                    it = pickle.loads(f.read())
                    write_iter(iter_dir, it.density_ratio, it.test_acc)
                    return it
                except:
                    gfile.Remove(plot_cache)

    execution_data_iter_dir = os.path.join(iter_dir.replace('results', 'execution_data'), 'eval')
    if not gfile.IsDirectory(execution_data_iter_dir):
        return None

    test_acc = None
    test_iter = None
    for events_file in gfile.ListDirectory(execution_data_iter_dir):
        if not events_file.startswith('events.out'):
            continue
        for e in tf.train.summary_iterator(os.path.join(execution_data_iter_dir, events_file)):
            for v in e.summary.value:
                if v.tag == 'accuracy' or v.tag == 'top_1_accuracy':
                    if test_iter is None or e.step > test_iter:
                        test_iter = e.step
                        test_acc = v.simple_value

    if verbose:
        print(test_acc)

    try:
        with gfile.Open(os.path.join(iter_dir, 'density_ratio')) as f:
            density_ratio = float(f.read())
    except Exception as e:
        density_ratio = 1.0

    res = IterDatum(
        iter=os.path.basename(iter_dir),
        density_ratio=density_ratio,
        test_acc=test_acc,
    )

    if can_write_cache and test_acc is not None:
        write_iter(iter_dir, density_ratio, test_acc)
        # with gfile.Open(plot_cache, 'w') as f:
        #     f.write('')
        #     f.flush()
        # with gfile.Open(plot_cache, 'wb') as f:
        #     pickle.dump(res, f)
    return res

def trial_datum_of_trial(experiment_dir, trial):
    plot_cache = os.path.join(experiment_dir, trial, 'plot_cache.pkl')
    if gfile.Exists(plot_cache):
        with gfile.Open(plot_cache, 'rb') as f:
            return pickle.loads(f.read())

    iter_dirs = sorted(iter_dirs_of_trial_dir(os.path.join(experiment_dir, trial)), key=lambda x:int(iter_re.match(os.path.basename(x)).group('iter')))

    pool = mp.Pool(5)

    res = TrialDatum(
        trial=trial,
        iter_data=list(filter(lambda x: x is not None, pool.map(iter_datum_of_iter_dir, map(lambda x: (x[1], x[0] < len(iter_dirs) - 1), enumerate(iter_dirs))))),
    )
    pool.close()
    return res

def get_experiment(experiment_dir, trials):
    return Experiment(
        experiment_dir=experiment_dir,
        trial_data=list(trial_datum_of_trial(experiment_dir, trial) for trial in trials),
    )

def experiments_of_directories(directories):
    trials = set(collect_trials(directories))
    if not trials:
        raise ValueError('No trials found for {}!'.format(directories))

    experiment_dir_to_trial_map = collections.defaultdict(list)
    for t in trials:
        experiment_dir_to_trial_map[os.path.dirname(t)].append(os.path.basename(t))

    return list(get_experiment(*x) for x in experiment_dir_to_trial_map.items())
