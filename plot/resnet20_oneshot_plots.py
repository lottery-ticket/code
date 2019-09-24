#!/usr/bin/env python3

import collections
import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
import plot_utils

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
conn = sqlite3.connect(os.path.join(_DIRNAME, 'iters.db'))

other = conn.execute('select * from iters where path like "%resnet20/prune_global_49/v5/finetune/finetune_130/trial_%/iter_0%"')
trial_map = {
    path.split('/')[-2]: acc
    for (path, _, acc) in other.fetchall()
}

cur = conn.execute('SELECT * FROM iters where path like "%resnet20/prune_global_%/v1%"')

# iter -> expt name -> point -> trial -> delta
iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))

for (name, density, acc) in cur.fetchall():
    if 'oneshot' not in name and 'reinit' not in name:
        continue
    if 'early' in name or 'short' in name:
        continue
    trial = name.split('/')[-2]
    base_acc = trial_map[trial]

    it = name.split('/')[-6]
    expt_name = name.split('/')[-4]
    point = int(name.split('/')[-3].split('_')[-1])

    if 'lottery' in expt_name:
        point = int(round((71108 - point) / (50000/128)))

    it = 'prune_{:.2f}'.format((1 - density) * 100)

    delta = acc - base_acc

    iters[it][expt_name][point][trial] = max(iters[it][expt_name][point].get(trial, delta), delta)

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
    elif 'reinit' in expt:
        return 'Reinitialize'
    else:
        print(expt)
        raise ValueError()




import tabulate
ft_means = []
ft_stds = []
lot_means = []
lot_stds = []
row = []
q = []

for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):
    real_it = str(it)
    row.append(real_it)
    ft_mean = []
    lot_mean = []
    ft_std = []
    lot_std = []
    q = sorted(expts['oneshot_finetune'].keys())
    q = [x for x in q if x <= 90]
    for i in q:
        ft_mean.append(np.mean(list(expts['oneshot_finetune'][i].values())))
        lot_mean.append(np.mean(list(expts['oneshot_lottery'][i].values())))
        ft_std.append(np.std(list(expts['oneshot_finetune'][i].values())))
        lot_std.append(np.std(list(expts['oneshot_lottery'][i].values())))
    ft_means.append(ft_mean)
    ft_stds.append(ft_std)
    lot_means.append(lot_mean)
    lot_stds.append(lot_std)


for (r, ls, fs) in zip(row, lot_means, ft_means):
    print(r)

    try:
        i = next(i for i in range(len(ls)) if ls[i] >= 0)
        print('Lot: {}'.format(q[i]))
    except StopIteration:
        print('Lot: N/A')

    try:
        i = next(i for i in range(len(ls)) if fs[i] >= 0)
        print('FT: {}'.format(q[i]))
    except StopIteration:
        print('FT: N/A')

    print('='*80)

import sys; sys.exit(0)




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
            xs.append(point)
            y_means.append(np.mean(deltas))
            y_errs.append(np.std(deltas) if len(deltas) > 1 else 0)

        plt.errorbar(xs, y_means, y_errs, fmt='o--', label=labof(expt) + ' mean $\pm 1$ std', color=cmap[expt.lstrip('oneshot_')])

    plt.plot([0, 71108/(50000/128)], [0, 0], '--', color=(0,0,0,0.3))
    plt.ylim(-0.03, 0.01)
    plt.xlabel('Re-training Epochs')
    plt.ylabel('$\Delta$ test accuracy')
    plt.legend()
    plt.title('ResNet-20 $\Delta$ test accuracy after one-shot prune to ${}\%$ sparsity'.format(it.split('_')[-1]))
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    ticks1 = points
    # ticks1 = np.linspace(0, 182, 10)
    int_ticks1 = [round(i) for i in ticks1]
    plt.gca().set_xticks(int_ticks1)

    plt.tight_layout()

    if plot_utils.save():
        plt.savefig(os.path.join('results_oneshot', 'resnet20_' + it + '.png'))

if plot_utils.show():
    plt.show()
