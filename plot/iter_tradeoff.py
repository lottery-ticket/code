#!/usr/bin/env python3

import argparse
import os
import re
import numpy as np
from utils import *

import gfile

type_re = re.compile(r'^(finetune_(?P<ft_idx>\d+)|prune_(?P<prune_idx>\d+))$')

def _get_iter(expt, trial, iter_idx):
    if isinstance(trial.iter_data, list):
        iter_data = trial.iter_data
    else:
        iter_data = trial.iter_data.get()

    iter_data = list(filter(lambda x: x is not None, iter_data))

    # if len(iter_data) == 21:
    #     plot_cache = os.path.join(expt.experiment_dir, trial.trial, 'plot_cache.pkl')
    #     with gfile.Open(plot_cache, 'wb') as f:
    #         pickle.dump(TrialDatum(trial.trial, iter_data), f)

    return next(i for i in iter_data if i.iter == 'iter_{}'.format(iter_idx))

def plot_experiments(expts, iter_idx):
    assert iter_idx > 0

    for iter_idx in range(1, 11, 1):
        prune_expts = {}
        ft_expts = {}

        for expt in expts:
            m = type_re.match(os.path.basename(expt.experiment_dir.rstrip('/')))
            accs = lambda it: np.array([_get_iter(expt, t, it).test_acc for t in expt.trial_data])

            try:
                deltas = accs(iter_idx) - accs(iter_idx-1)
            except StopIteration:
                continue

            mean = deltas.mean()
            err_below = deltas.mean() - deltas.min()
            err_above = deltas.max() - deltas.mean()

            if m.group('ft_idx'):
                ft_expts[int(m.group('ft_idx'))] = [mean, err_below, err_above]
            elif m.group('prune_idx'):
                prune_expts[int(182 - round(int(m.group('prune_idx')) / 391.0))] = [mean, err_below, err_above]

        plt.figure()

        plt.title('Iter {}'.format(iter_idx))

        xs, ys = zip(*sorted(list(prune_expts.items())))
        means, below, above = zip(*ys)
        plt.errorbar(xs, means, fmt='o--', yerr=np.stack((below, above)), label='Lottery')

        xs, ys = zip(*sorted(list(ft_expts.items())))
        means, below, above = zip(*ys)
        plt.errorbar(xs, means, fmt='o-', yerr=np.stack((below, above)), label='Finetune')

        plt.legend()

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='+')
    parser.add_argument('--iter', type=int, required=True)
    args = parser.parse_args()

    plot_experiments([expt for directory in args.directory for expt in experiments_of_directory(directory)], args.iter)


if __name__ == '__main__':
    main()
