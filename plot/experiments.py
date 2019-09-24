#!/usr/bin/env python3

import argparse
import collections
import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from utils import *

def plot_experiments(experiments):
    plt.xlim(1, 0)

    to_plot = []

    for i, experiment in enumerate(experiments):
        name = experiment.experiment_dir

        all_xs = collections.defaultdict(list)
        all_ys = collections.defaultdict(list)

        for trial in experiment.trial_data:
            if isinstance(trial.iter_data, list):
                iter_data = trial.iter_data
            else:
                iter_data = trial.iter_data.get()
            iter_data = [it for it in iter_data if it is not None]
            iter_data = sorted(iter_data, key=lambda x: x.density_ratio)

            if len(iter_data) == 21:
                plot_cache = os.path.join(name, trial.trial, 'plot_cache.pkl')
                with tf.gfile.Open(plot_cache, 'wb') as f:
                    pickle.dump(TrialDatum(trial.trial, iter_data), f)

            for it in iter_data:
                iter_idx = int(iter_re.match(it.iter).group('iter'))
                all_xs[iter_idx].append(it.density_ratio)
                all_ys[iter_idx].append(it.test_acc)

        xs_iter = [x[1] for x in sorted(all_xs.items())]
        ys_iter = [y[1] for y in sorted(all_ys.items())]


        xs = np.array([sum(x)/len(x) for x in xs_iter])
        ys_mean = np.array([sum(y)/len(y) for y in ys_iter])
        ys_min = np.array([min(y) for y in ys_iter])
        ys_max = np.array([max(y) for y in ys_iter])
        yerr = np.stack((ys_min - ys_mean, ys_mean - ys_max))

        if len(xs) < 18:
            continue

        to_plot.append((xs, ys_mean, yerr, name))

    cmap = plt.get_cmap('tab20')
    for i, (xs, ys_mean, yerr, name) in enumerate(sorted(to_plot, key=lambda x: -x[1].mean())):
        plt.errorbar(
            xs,
            ys_mean,
            yerr=yerr,
            fmt='o--',
            color=cmap(i/len(to_plot)),
            label=name,
        )

    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', nargs='+')
    args = parser.parse_args()

    experiments = [expt for directory in args.directory for expt in experiments_of_directory(directory)]
    plot_experiments(experiments)

if __name__ == '__main__':
    main()
