#!/usr/bin/env python3

import collections
import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
import plot_utils

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
conn = sqlite3.connect(os.path.join(_DIRNAME, 'iters.db'))

cur = conn.execute('SELECT * FROM iters where path like "%resnet20/global_magnitude_20/v2%"')

all_experiments = cur.fetchall()

# iter -> expt name -> point -> trial -> delta
iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
it_density_dict = collections.defaultdict(list)
legal_points = list(map(int, '0 130 70 170 10 110 50 150 182 90 30'.split()))

for (name, density, acc) in all_experiments:
    if 'early' in name or 'oneshot' in name or 'reinit' in name:
        continue

    trial = name.split('/')[-2]

    it = int(name.split('/')[-1].split('_')[-1])
    expt_name = name.split('/')[-4]
    point = int(name.split('/')[-3].split('_')[-1])

    if 'lottery' in expt_name:
        point = int(round((71108 - point) / (50000/128)))

    if point not in legal_points:
        continue

    it_density_dict[it].append(density)
    iters[it][expt_name][point][trial] = max(iters[it][expt_name][point].get(trial, acc), acc)

cmap = {
    'finetune': 'C2',
    'reinit': 'C3',
    'lottery': 'C1',
    'lottery_early_40': 'C0',
}

allowed = [
    '95.6',
    '94.5',
    '79.03',
    '36.0',
]

def labof(expt):
    if 'lottery' in expt:
        return 'Rewind-Replay'
    elif 'finetune' in expt:
        return 'Finetune'
    else:
        raise ValueError(expt)

def suffix_of_number(myDate):
    date_suffix = ["th", "st", "nd", "rd"]
    if myDate % 10 in [1, 2, 3] and myDate not in [11, 12, 13]:
        return date_suffix[myDate % 10]
    else:
        return date_suffix[0]

for (it, expts) in sorted(iters.items()):
    density = np.mean(it_density_dict[it])
    diff = np.max(it_density_dict[it]) - np.min(it_density_dict[it])
    print('{}: {} ({} - {})'.format(it, diff, np.min(it_density_dict[it]), np.max(it_density_dict[it])))

    plt.figure(figsize=(8,4))
    for (expt, points) in sorted(expts.items()):
        xs = []
        y_means = []
        y_errs = []
        for (point, delta_dict) in sorted(points.items()):
            try:
                deltas = [final - iters[0][expt][point][t] for (t, final) in delta_dict.items()]
            except:
                continue
            xs.append(point)
            y_means.append(np.mean(deltas))
            y_errs.append(np.std(deltas) if len(deltas) > 1 else 0)

        plt.errorbar(xs, y_means, y_errs, fmt='o--', label=labof(expt) + ' mean $\pm 1$ std', color=cmap[expt.lstrip('oneshot_')])

    plt.plot([0, 71108/(50000/128)], [0, 0], '--', color=(0,0,0,0.3))
    plt.ylim(-0.03, 0.01)
    plt.xlabel('Re-training Epochs')
    plt.ylabel('$\Delta$ test accuracy')
    plt.legend()
    plt.tight_layout()
    plt.title('ResNet-20 $\Delta$ test accuracy after {idx}{suffix} pruning iteration ({sparsity:.2%} sparsity)'.format(
        idx=it,
        suffix=suffix_of_number(it),
        sparsity=1-density,
    ))
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])


    ticks1 = points
    # ticks1 = np.linspace(0, 182, 10)
    int_ticks1 = [round(i) for i in ticks1]
    plt.gca().set_xticks(int_ticks1)

    vals = plt.gca().get_xticks()
    plt.gca().set_xticklabels([r'${} \times {}$'.format(int(x), it) if x > 0 else '0' for x in vals])

    plt.tight_layout()

    if plot_utils.save():
        plt.savefig(os.path.join('results_iterative', 'resnet20_{}'.format(it) + '.png'))

if plot_utils.show():
    plt.show()
