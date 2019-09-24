import tensorflow as tf
import functools
import sys

def _retry_unavailable(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            i = 1
            try:
                return func(*args, **kwargs)
            except tf.errors.UnavailableError:
                print('Got UnavailableError({}). Retrying...'.format(i), file=sys.stderr)
                i += 1
    return wrapper

@_retry_unavailable
def Exists(fname):
    return tf.gfile.Exists(fname)

@_retry_unavailable
def ListDirectory(fname):
    return tf.gfile.ListDirectory(fname)

@_retry_unavailable
def IsDirectory(fname):
    return tf.gfile.IsDirectory(fname)

@_retry_unavailable
def Open(fname, mode='r'):
    return tf.gfile.Open(fname, mode)

@_retry_unavailable
def Remove(fname):
    return tf.gfile.Remove(fname)
