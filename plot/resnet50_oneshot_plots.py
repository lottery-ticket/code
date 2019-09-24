#!/usr/bin/env python3

import collections
import sqlite3
import matplotlib.pyplot as plt
import os
import numpy as np
import plot_utils

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
conn = sqlite3.connect(os.path.join(_DIRNAME, 'iters.db'))

other = conn.execute('select * from iters where path like "%resnet50/prune_global_20/v10/finetune/finetune_9/trial_%/iter_0%"')
trial_map = {
    path.split('/')[-2]: acc
    for (path, _, acc) in other.fetchall()
}

cur = conn.execute('SELECT * FROM iters where path like "%resnet50%oneshot%"')

# iter -> expt name -> point -> trial -> delta
iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))

for (name, density, acc) in cur.fetchall():
    if 'oneshot' not in name and 'reinit' not in name:
        continue
    if 'short' in name or 'early' in name:
        continue
    trial = name.split('/')[-2]
    base_acc = trial_map[trial]

    it = name.split('/')[-1]
    if it.startswith('iter_'):
        continue

    expt_name = name.split('/')[-4]
    point = int(name.split('/')[-3].split('_')[-1])

    if 'lottery' in expt_name or 'reinit' in expt_name:
        point = int(round((112590 - point) / 1251))

    it = 'prune_{:.2f}'.format(density * 100)

    delta = acc - base_acc

    if abs(float(it.split('_')[-1]) / 100 - density) > 1e-3:
        raise ValueError((float(it.split('_')[-1]) / 100, density))

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
        print(expt)
        raise ValueError()




import tabulate
ft_means = []
ft_stds = []
lot_means = []
lot_stds = []
row = []
q = []

for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1]))[::-1]:
    real_it = '{:.2f}'.format(100 - float(it.split('_')[-1]))
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



res = []
for (rit, f_ms,f_ss,l_ms,l_ss) in zip(row, ft_means,ft_stds,lot_means,lot_stds):
    res.append([])
    for (f_m,f_s,l_m,l_s) in zip(f_ms,f_ss,l_ms,l_ss):
        # res[-1].append('{:.2%} ± {:.2%} / {:.2%} ± {:.2%}'.format(f_m,f_s,l_m,l_s))
        res[-1].append('{:+07.2%} | {:+07.2%}'.format(f_m,l_m))

fig, ax = plt.subplots()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


z = np.array(lot_means) - np.array(ft_means)
vmin = z.min()
vmax = z.max()

m_cmap = shiftedColorMap(matplotlib.cm.seismic, midpoint=(1 - vmax / (vmax + abs(vmin))), name='shifted')

im = plt.imshow(z, cmap=m_cmap)
ax.set_xticks(np.arange(len(q)))
ax.set_xticklabels(list(map('{} epochs'.format, q)))
ax.set_yticks(np.arange(len(z)))
ax.set_yticklabels(list(map('{}% sparsity'.format, row)))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel(r'$\Delta$ accuracy (lot - ft)')
plt.tight_layout()
plt.show()


print(tabulate.tabulate(res, headers=['Sparsity', '9 re-train epochs', '45 re-train epochs', '81 re-train epochs']))

import sys
sys.exit(0)






for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):
    real_it = '{:.2f}'.format(100 - float(it.split('_')[-1]))

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
            center = np.mean(deltas)
            y_means.append(center)
            y_errs.append(np.std(deltas) if len(deltas) > 1 else 0)
            # y_errs.append(np.abs((np.array([np.min(deltas), np.max(deltas)]) - center)))

        # y_errs = np.moveaxis(y_errs, 1, 0)
        plt.errorbar(xs, y_means, y_errs, fmt='o--', label=labof(expt) + ' mean $\pm 1$ std', color=cmap[expt.lstrip('oneshot_')])

    plt.plot([0, 90], [0, 0], '--', color=(0,0,0,0.3))
    plt.ylim(-0.03, 0.01)
    plt.xlabel('Re-training Epochs')
    plt.ylabel('$\Delta$ test accuracy')
    plt.legend()
    plt.title('ResNet-50 $\Delta$ test accuracy after one-shot prune to ${}\%$ sparsity'.format(real_it))
    vals = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    ticks1 = np.linspace(0,90,11)
    int_ticks1 = [round(i) for i in ticks1]
    plt.gca().set_xticks(int_ticks1)

    plt.tight_layout()

    if plot_utils.save():
        plt.savefig(os.path.join('results_oneshot', 'resnet50_{}'.format(real_it) + '.png'))

if plot_utils.show():
    plt.show()
