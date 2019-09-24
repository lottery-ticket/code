#!/usr/bin/env python3

import collections
import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
import plot_utils

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
conn = sqlite3.connect(os.path.join(_DIRNAME, 'iters.db'))

other = conn.execute('select * from iters where path like "s3://REDACTED-data/results/vgg_16_nofc/prune_global_20_fc/v2/lottery/prune_4701/trial_%/iter_0"')
trial_map = {
    path.split('/')[-2]: acc
    for (path, _, acc) in other.fetchall()
}

cur = conn.execute('SELECT * FROM iters where path like "%vgg_16_nofc/prune_global_%/v4%"')

# iter -> expt name -> point -> trial -> delta
iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))

for (name, density, acc) in cur.fetchall():
    if 'oneshot' not in name and 'reinit' not in name:
        continue
    if 'lottery_early_40' in name:
        continue
    trial = name.split('/')[-2]
    base_acc = trial_map[trial]

    it = name.split('/')[-6]
    expt_name = name.split('/')[-4]
    point = int(name.split('/')[-3].split('_')[-1])

    if 'lottery' in expt_name:
        point = int(round((71108 - point) / (50000/128)))

    delta = acc - base_acc

    it = 'prune_{:.2f}'.format((1 - density) * 100)

    # if point == 110 and 'lottery' in expt_name and '36.0' in name:
    #     print('{} was {} now {}'.format(name, iters[it][expt_name][point].get(trial, 'na'), max(iters[it][expt_name][point].get(trial, delta), delta)))

    iters[it][expt_name][point][trial] = max(iters[it][expt_name][point].get(trial, delta), delta)

cmap = {
    'finetune': 'C2',
    'reinit': 'C3',
    'lottery': 'C1',
    'lottery_early_40': 'C0',
}

allowed = [
    '98.56',
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
    elif 'reinit' in expt:
        return 'Reinitialize'
    else:
        raise ValueError()

for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):

    if False and not any(a in it for a in allowed):
        continue

    plt.figure(figsize=(8,4))
    for (expt, points) in sorted(expts.items()):
        xs = []
        y_means = []
        y_errs = []
        for (point, delta_dict) in sorted(points.items()):
            deltas = list(delta_dict.values())

            if point == 110 and 'lottery' in expt and '36.0' in it:
                print('{} was {} now {}'.format(it, iters[it][expt][point].get(trial, 'na'), max(iters[it][expt][point].get(trial, delta), delta)))
                print(deltas)

            xs.append(point)
            y_means.append(np.mean(deltas))
            y_errs.append(np.std(deltas) if len(deltas) > 1 else 0)

        plt.errorbar(xs, y_means, y_errs, fmt='o--', label=labof(expt) + ' mean $\pm 1$ std', color=cmap[expt.lstrip('oneshot_')])

    plt.plot([0, 71108/(50000/128)], [0, 0], '--', color=(0,0,0,0.3))
    plt.ylim(-0.03, 0.01)
    plt.xlabel('Re-training Epochs')
    plt.ylabel('$\Delta$ test accuracy')
    plt.legend()
    plt.title('VGG-16 $\Delta$ test accuracy after one-shot prune to ${}\%$ sparsity'.format(it.split('_')[-1]))
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    ticks1 = points
    # ticks1 = np.linspace(0, 182, 10)
    int_ticks1 = [round(i) for i in ticks1]
    plt.gca().set_xticks(int_ticks1)

    plt.tight_layout()

    if plot_utils.save():
        plt.savefig(os.path.join('results_oneshot', 'vgg16_' + it + '.png'))

if plot_utils.show():
    plt.show()
