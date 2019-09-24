#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import argparse
import functools
import gfile
import multiprocessing
import os
import sqlite3
import tensorflow as tf
import threading
import traceback

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
_SQLITE_FILE = os.path.join(_DIRNAME, 'data.db')

if not os.path.exists(_SQLITE_FILE):
    with sqlite3.connect(_SQLITE_FILE) as conn:
        conn.execute("CREATE TABLE data (path TEXT NOT NULL, event_file TEXT NOT NULL, step INT NOT NULL, density REAL, test_acc REAL NOT NULL, PRIMARY KEY (path, event_file))")

def _try_until_success(func):
    @functools.wraps(func)
    def tryit(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except:
                traceback.print_exc()
    return tryit

@_try_until_success
def write_datum(path, event_file, step, density_ratio, test_acc):
    with sqlite3.connect(_SQLITE_FILE) as conn:
        conn.execute(
            'REPLACE INTO data(path,event_file,step,density,test_acc) VALUES (?,?,?,?,?)',
            (path.rstrip('/'), event_file, step, density_ratio, test_acc)
        )

@_try_until_success
def read_latest_datum(path):
    with sqlite3.connect(_SQLITE_FILE) as conn:
        cur = conn.execute(
            'SELECT path,step,density,test_acc FROM iters WHERE path=? ORDER BY -step LIMIT 1',
            (path.rstrip('/'),)
        )
        return cur.fetchone()

@_try_until_success
def read_datum_at_step(path, step):
    with sqlite3.connect(_SQLITE_FILE) as conn:
        cur = conn.execute(
            'SELECT path,step,density,test_acc FROM iters WHERE path=? AND step=? LIMIT 1',
            (path.rstrip('/'), step)
        )
        return cur.fetchone()


_N_WORKERS = 32

def _scan(to_scan_q, to_read_q):
    while True:
        path = to_scan_q.get()
        try:
            if path is None:
                return

            for f in gfile.ListDirectory(path):
                f = os.path.join(path, f)
                if 'eval/events.out.tfevents' in f:
                    to_read_q.put(f)
                elif f.endswith('/') or (f.startswith('s3://') and ('model.ckpt' not in f) and gfile.IsDirectory(f)):
                    to_scan_q.put(f)
        except:
            traceback.print_exc()
        finally:
            to_scan_q.task_done()
        # print('scanned {} in {} seconds'.format(path, time.time() - start))

def _read(path):
    test_acc = None
    test_iter = None
    for e in tf.data.TFRecordDataset(path):
        e = tf.Event.FromString(e.numpy())
        for v in e.summary.value:
            if v.tag == 'accuracy' or v.tag == 'top_1_accuracy':
                if test_iter is None or e.step > test_iter:
                    test_iter = e.step
                    test_acc = v.simple_value

    density_ratio = float('nan')
    path = path.split('/')
    path, event_file = path[:-1], path[-1]
    path = '/'.join(path)

    if test_iter is not None:
        write_datum(path, event_file, test_iter, density_ratio, test_acc)

def read_all(*paths):
    m = multiprocessing.Manager()
    to_scan_q = m.JoinableQueue()
    to_read_q = m.JoinableQueue()

    def scan_manager():
        for path in paths:
            to_scan_q.put(path)
        p = multiprocessing.Pool()
        p.starmap_async(_scan, [(to_scan_q, to_read_q) for _ in range(_N_WORKERS)])
        to_scan_q.join()
        for _ in range(_N_WORKERS):
            to_scan_q.put(None)
        p.close()

    i = 0

    def read_manager_single():
        nonlocal i
        while True:
            i += 1
            path = to_read_q.get()
            # if i % 100 == 0:
            print(i, end='\r')
            try:
                if path is None:
                    return
                p = multiprocessing.Process(target=_read, args=(path,))
                p.start()
                p.join()
            finally:
                to_read_q.task_done()

    def read_manager():
        ts = []
        for _ in range(_N_WORKERS):
            ts.append(threading.Thread(target=read_manager_single))
            ts[-1].start()
        to_scan_q.join()
        for _ in range(_N_WORKERS):
            to_read_q.put(None)
        for t in ts:
            t.join()
        to_read_q.join()

    smt = threading.Thread(target=scan_manager)
    smt.start()
    rmt = threading.Thread(target=read_manager)
    rmt.start()

    smt.join()
    rmt.join()

def main():
    multiprocessing.set_start_method('forkserver')
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='+')
    args = parser.parse_args()
    print(args.path)
    read_all(*args.path)

if __name__ == '__main__':
    main()
