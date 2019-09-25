#!/usr/bin/env python3

import atexit
import collections
import iters
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import multiprocessing
import numpy as np
import operator
import os
import pandas as pd
import random
import scipy.stats
import shutil
import sqlite3
import sys
import tempfile
import contextlib
from matplotlib.colors import ListedColormap
from matplotlib.cm import register_cmap

_DIRNAME = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))
conn = sqlite3.connect(os.path.join(_DIRNAME, 'data.db'))

FIG_DPI = 100

MEDIAN_MAX = 'max'
MEAN_STD = 'mean'

RANGE_METHOD = MEDIAN_MAX

if RANGE_METHOD == MEDIAN_MAX:
    _FLAVOR = 'median $\pm$ min/max'
elif RANGE_METHOD == MEAN_STD:
    _FLAVOR = 'mean $\pm 1$ std'
else:
    raise ValueError(RANGE_METHOD)


class SparsityFormatter(ticker.PercentFormatter):
    def __init__(self, *args, flip=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.flip = flip

    def __call__(self, x, i=None):
        if self.flip:
            x = 1 - x
        return super().__call__(x, i)

class MidpointNormalize(matplotlib.colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def argcummax(x):
    if len(x) == 0:
        return []
    maxidx = 0
    maxval = x[0]
    res = [maxidx]
    for i in range(1, len(x)):
        if x[i] > maxval:
            maxidx = i
            maxval = x[i]
        res.append(maxidx)
    return res

def fix_deltas_base(iters, base_url):
    # base_url like 'resnet20/prune_global_49/v5/finetune/finetune_130'
    com = 'select * from data where (path, step) in (select path, max(step) from data where path like "%{}/trial_%/iter_0%" group by path)'.format(base_url)
    cur = conn.execute(com)
    base_map = {
        path.split('/')[-3]: acc
        for (path, _, step, _, acc) in cur.fetchall()
    }

    for it, expts in iters.items():
        for expt, points in expts.items():
            for point, trials in points.items():
                for t in trials:
                    try:
                        trials[t] -= base_map[t]
                    except KeyError:
                        import ipdb; ipdb.set_trace()

def fix_deltas_iter(iters):
    for it, expts in iters.items():
        if it == 'iter_0':
            continue
        for expt, points in expts.items():
            for point, trials in points.items():
                for t in trials:
                    trials[t] -= iters['iter_0'][expt][point][t]

def get_long_iter_dict(base_url, allowed_names, iterations_per_epoch, epochs):
    # base_url like 'resnet20/prune_global_%/v1'
    # allowed_names like ['lottery', 'oneshot_lottery', 'reinit']

    allowed_names = ['/{}/'.format(name.strip('/')) for name in allowed_names]
    # iter -> expt name -> point -> trial -> delta
    iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    com = 'select path, step, test_acc from data where path like "%{}%"'.format(base_url)
    cur = conn.execute(com)

    for (name, step, acc) in cur.fetchall():
        name = '/'.join(name.split('/')[:-1])
        if not any(allowed_name in name for allowed_name in allowed_names):
            continue

        trial = name.split('/')[-2]
        if trial not in ('trial_1', 'trial_2', 'trial_3'):
            continue

        expt_name = name.split('/')[-4]
        iter_int = int(name.split('/')[-3].split('_')[-1])
        it = 'iter_{}'.format(iter_int)
        if it == 'iter_0':
            continue
        point = round((step - iterations_per_epoch * epochs) / (iterations_per_epoch * iter_int))

        try:
            prev = iters[it][expt_name][point][trial]
        except KeyError:
            iters[it][expt_name][point][trial] = acc

    return iters

def merge_long_reinits(iters):
    res = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    reinit = 'reinit_long'
    for i in iters:
        for e in iters[i]:
            for p in iters[i][e]:
                for t in iters[i][e][p]:
                    a = iters[i][e][p][t]
                    d = res[i][reinit][p]
                    if t in d:
                        d[t] = max(a, d[t])
                    else:
                        d[t] = a
    return res


def fix_long_deltas(iters, network):
    network_long_bases = {
        'resnet20': '%resnet20/prune_global_20/v3/lottery/prune_%/trial_%/iter_%',
        'resnet50': '%resnet50/prune_global_20/v10/lottery/prune_11259/trial_%/iter_%',
        'vgg16': '%vgg_16_nofc/prune_global_20/v6/base/trial_%/iter_%',
    }
    network_iter_bases = {
        'resnet20': [4701, 4701, 4701, 28139, 4701, 12514, 12514, 12514, 12514, 12514],
        'resnet50': [11259] * 10,
        'vgg16': [0] * 30,
    }
    com = 'select path,test_acc from data where (path, step) in (select path, max(step) from data where path like "{}" group by path)'.format(network_long_bases[network])
    cur = conn.execute(com)
    res_dict = {}
    for path,test_acc in cur.fetchall():

        if 'resnet' in network:
            prune = int(path.split('/')[-4].split('_')[-1])
            trial = path.split('/')[-3]
            it = int(path.split('/')[-2].split('_')[-1])
            res_dict[prune,trial,it] = test_acc
        else:
            prune = 0
            trial = path.split('/')[-3]
            it = int(path.split('/')[-2].split('_')[-1])
            res_dict[prune,trial,it] = test_acc

    for it_s in iters:
        it = int(it_s.split('_')[-1]) - 1
        for expt in iters[it_s]:

            if 'best' in expt:
                m_it = it
            elif 'oneshot' in expt:
                m_it = 0
            else:
                raise ValueError()

            for point in iters[it_s][expt]:
                for trial in iters[it_s][expt][point]:
                    iters[it_s][expt][point][trial] -= res_dict[network_iter_bases[network][it],trial,m_it]


def get_iter_dict(base_url, allowed_names, iterations_per_epoch, epochs, it_idx):
    # base_url like 'resnet20/prune_global_%/v1'
    # allowed_names like ['lottery', 'oneshot_lottery', 'reinit']

    allowed_names = ['/{}/'.format(name.strip('/')) for name in allowed_names]

    # iter -> expt name -> point -> trial -> delta
    iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    com = 'select * from data where (path, step) in (select path, max(step) from data where path like "%{}%" group by path)'.format(base_url)
    cur = conn.execute(com)

    for (name, _, step, _, acc) in cur.fetchall():
        name = '/'.join(name.split('/')[:-1])
        if not any(allowed_name in name for allowed_name in allowed_names):
            continue

        trial = name.split('/')[-2]
        if trial not in ('trial_1', 'trial_2', 'trial_3'):
            continue

        expt_name = name.split('/')[-4]
        point = int(name.split('/')[-3].split('_')[-1])
        it = name.split('/')[it_idx]

        if 'lottery' in expt_name:
            point = int(round((epochs * iterations_per_epoch - point) / iterations_per_epoch))
        elif 'real_reinit' in expt_name:
            point = int(point / iterations_per_epoch)

        try:
            prev = iters[it][expt_name][point][trial]
        except KeyError:
            iters[it][expt_name][point][trial] = acc

    return iters

def merge(i1, i2):
    iters = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for i in [i1,i2]:
        for a in i:
            for b in i[a]:
                for c in i[a][b]:
                    for d in i[a][b][c]:
                        iters[a][b][c][d] = i[a][b][c][d]
    return iters

def resnet20_oneshot_deltas():
    iters = get_iter_dict('resnet20/prune_global_%/v4', ['oneshot_lottery', 'oneshot_finetune', 'oneshot_real_reinit'], 50000/128, 182, -1)
    fix_deltas_base(iters, 'resnet20/prune_global_20/v3/finetune/finetune_90')
    return iters

def resnet20_iterative_deltas():
    iters = get_iter_dict('resnet20/prune_global_20/v3', ['lottery', 'finetune'], 50000/128, 182, -1)
    fix_deltas_iter(iters)
    long_it = get_long_iter_dict('%resnet20%v4%', ['reinit_best_long', 'reinit_oneshot_long'], 50000/128, 182)
    fix_long_deltas(long_it, 'resnet20')
    long_it = merge_long_reinits(long_it)
    res = merge(iters, long_it)
    remove_other_points(0, 10, 30, 50, 70, 90, 110, 130, 150, 170, 182, 192, 212, 232, 252, 272, 292, 312, 332, 352, 364)(res)
    return res

def resnet50_oneshot_deltas():
    iters = get_iter_dict('resnet50/prune_global_%/v1%', ['oneshot_lottery', 'oneshot_finetune', 'oneshot_real_reinit'], 1251, 90, -1)
    fix_deltas_base(iters, 'resnet50/prune_global_20/v10/finetune/finetune_9')
    return iters

def resnet50_iterative_deltas():
    iters = get_iter_dict('resnet50%v10', ['lottery', 'finetune'], 1251, 90, -1)
    fix_deltas_iter(iters)
    long_it = get_long_iter_dict('%resnet50%v10%', ['reinit_best_long', 'reinit_oneshot_long'], 1251, 90)
    fix_long_deltas(long_it, 'resnet50')
    long_it = merge_long_reinits(long_it)
    res = merge(iters, long_it)
    remove_other_points(*range(0, 181, 9))(res)
    return res

def vgg16_oneshot_deltas():
    iters = get_iter_dict('vgg_16_nofc/prune_global_20/v8', ['oneshot_lottery', 'oneshot_finetune', 'oneshot_reinit'], 50000/128, 182, -1)
    fix_deltas_base(iters, 's3://REDACTED-data/execution_data/vgg_16_nofc/prune_global_20/v6/base')
    return iters

def vgg16_iterative_deltas():
    iters = get_iter_dict('vgg_16_nofc/prune_global_20/v8', ['lottery', 'finetune'], 50000/128, 182, -1)
    fix_deltas_iter(iters)
    long_it = get_long_iter_dict('%vgg_16_nofc/prune_global_20/v6%', ['reinit_best_long', 'reinit_oneshot_long'], 50000/128, 182)
    fix_long_deltas(long_it, 'vgg16')
    long_it = merge_long_reinits(long_it)
    res = merge(iters, long_it)
    remove_other_points(0, 10, 30, 50, 70, 90, 110, 130, 150, 170, 182, 192, 212, 232, 252, 272, 292, 312, 332, 352, 364)(res)
    return res

def vgg19_oneshot_deltas():
    iters = get_iter_dict('vgg_19_nofc/prune_global_%/v8', ['oneshot_lottery', 'oneshot_finetune', 'oneshot_reinit'], 50000/128, 182, -1)
    fix_deltas_base(iters, 's3://REDACTED-data/execution_data/vgg_19_nofc/prune_global_20/v6/base')
    return iters

def vgg19_iterative_deltas():
    # iters = get_iter_dict('vgg_19_nofc/prune_global_20/v8', ['lottery', 'finetune'], 50000/128, 182, -1)
    # fix_deltas_iter(iters)
    iters = get_iter_dict('vgg_19_nofc/prune_global_20_fc/v3', ['lottery', 'finetune'], 50000/128, 182, -1)
    fix_deltas_iter(iters)
    long_it = get_long_iter_dict('%vgg_19_nofc/prune_global_20/v6%', ['reinit_best_long', 'reinit_oneshot_long'], 50000/128, 182)
    fix_long_deltas(long_it, 'vgg19')
    long_it = merge_long_reinits(long_it)
    res = merge(iters, long_it)
    remove_other_points(0, 10, 30, 50, 70, 90, 110, 130, 150, 170, 182, 192, 212, 232, 252, 272, 292, 312, 332, 352, 364)(res)
    return res

def get_color(name):
    def c(*col):
        return (col[0]/255, col[1]/255, col[2]/255, 1)
    cmap = {
        'lottery': '#1b9e77',
        'finetune': '#d95f02',
        'reinit': '#7570b3',
    }
    return next(cmap[x] for x in cmap if x in name)

def labof(expt):
    if 'lottery' in expt:
        return 'Rewind-Replay'
    elif 'finetune' in expt:
        return 'Fine-tune'
    elif 'reinit' in expt:
        return 'Reinitialize'
    else:
        raise ValueError(expt)

def fmtof(expt):
    if 'lottery' in expt:
        return 'o--'
    elif 'finetune' in expt:
        return '^:'
    elif 'reinit' in expt:
        return 'x-.'
    else:
        raise ValueError(expt)


def suffix_of_number(myDate):
    date_suffix = ["th", "st", "nd", "rd"]
    if myDate % 10 in [1, 2, 3] and myDate not in [11, 12, 13]:
        return date_suffix[myDate % 10]
    else:
        return date_suffix[0]

def legend_sort_key(x):
    if 'lottery' in x.lower():
        return 0
    elif 'finetune' in x.lower():
        return 1
    elif 'reinit' in x.lower():
        return 2
    else:
        raise ValueError()

def plot_iters(iters, name, is_iterative, show=True, savename=None, maxit=True):
    if show:
        plt.ion()
    else:
        plt.ioff()

    xs_map = {}
    for expt, expts in iters.items():
        xs_map[expt] = set()
        for points in expts.values():
            xs_map[expt] = xs_map[expt].union(set(points.keys()))

    all_xs = set.union(*xs_map.values())

    for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):
        if it == 'iter_0':
            continue
        elif it == 'iter_1' and name == 'resnet50' and not is_iterative:
            continue

        plt.figure()
        for (expt, points) in sorted(expts.items(), key=lambda x: legend_sort_key(x[0])):
            xs = []
            y_means = []
            y_errs_lower = []
            y_errs_upper = []
            for (point, delta_dict) in sorted(points.items()):
                deltas = list(delta_dict.values())
                xs.append(point)

                if RANGE_METHOD == MEDIAN_MAX:
                    y_means.append(np.median(deltas))
                    y_errs_lower.append(y_means[-1] - np.min(deltas))
                    y_errs_upper.append(np.max(deltas) - y_means[-1])
                elif RANGE_METHOD == MEAN_STD:
                    y_means.append(np.mean(deltas))
                    std = np.std(deltas) if len(deltas) > 1 else 0
                    if std > 0.03:
                        std = 0
                    y_errs_lower.append(std)
                    y_errs_upper.append(std)
                else:
                    raise ValueError(RANGE_METHOD)

            xs = np.array(xs)

            y_means = np.array(y_means)
            y_errs_lower = np.array(y_errs_lower)
            y_errs_upper = np.array(y_errs_upper)

            idxs = [i for i in range(len(xs)) if xs[i] in all_xs]
            max_idxs = argcummax(y_means[idxs])

            plt.errorbar(xs[idxs], y_means[idxs],
                         [y_errs_lower[idxs], y_errs_upper[idxs]],
                         fmt=fmtof(expt), label='{} {}'.format(labof(expt), _FLAVOR),
                         color=get_color(expt),
                         capsize=5,
            )

        if name == 'resnet50':
            lr_changes = [10, 30, 60]
        else:
            lr_changes = [46, 91]

        # for ch in lr_changes:
        #     plt.plot([ch, ch], [-0.03, 0.01], '--', color=(0,0,0,0.3))

        plt.plot([0, max(all_xs)], [0, 0], '--', color=(0,0,0,0.3))
        plt.ylim(-0.03, 0.01)
        if is_iterative:
            plt.xlabel(r'Re-training Epochs (epochs per iteration $\times$ iterations)')
        else:
            plt.xlabel('Re-training Epochs')
        plt.ylabel('$\Delta$ Accuracy')
        plt.legend()

        if is_iterative:
            it = int(it.split('_')[1])
            density = 0.8 ** it
            plt.title('{name} $\Delta$ accuracy after {idx}{suffix} pruning iteration ({sparsity:.2%} sparsity)'.format(
                name=netname(name),
                idx=it,
                suffix=suffix_of_number(it),
                sparsity=1-density,
            ).replace('%', r'\%'))
        else:
            density = float(it.split('_')[1]) / 100
            plt.title('{name} $\Delta$ accuracy after one-shot prune to {sparsity:.2%} sparsity'.format(
                name=netname(name),
                sparsity=1-density,
            ).replace('%', r'\%'))

        plt.tight_layout()
        format_axes(plt.gca())

        plt.gca().set_xticks([round(i) for i in sorted(all_xs)])
        vals = plt.gca().get_xticks()
        if is_iterative:
            plt.gca().set_xticklabels([r'${} \times {}$'.format(int(x), it) if x > 0 else '0' for x in vals], rotation=30, ha='right')


        vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:+,.2%}'.format(x).replace('%', r'\%') for x in vals])

        if savename:
            os.makedirs(os.path.join(_DIRNAME, 'figures', savename), exist_ok=True)
            plt.savefig(os.path.join(_DIRNAME, 'figures', savename, '{:.5}.pdf'.format(1 - density)), dpi=FIG_DPI)

        if not show:
            plt.close('all')

    if show:
        plt.show()

def diverging_colormap(x):
    from numpy import array
    x = array(x)
    if any(x < 0) or any(x > 1):
        raise ValueError('x must be between 0 and 1 inclusive.')
    red = (0.237 - 2.13 * x + 26.92 * x ** 2 - 65.5 * x ** 3 +
           63.5 * x ** 4 - 22.36 * x ** 5)
    grn = ((0.572 + 1.524 * x - 1.811 * x ** 2) /
           (1 - 0.291 * x + 0.1574 * x ** 2)) ** 2
    blu = 1. / (1.579 - 4.03 * x + 12.92 * x ** 2 - 31.4 * x ** 3 +
                48.6 * x ** 4 - 23.36 * x ** 5)
    return array([red, grn, blu]).T

register_cmap(cmap=ListedColormap(
    diverging_colormap(np.linspace(0, 1, 256)),
    name='custom_diverging'))

def usability_implot(iters, name, is_iterative, show=True, savename=None,
                     bars_to_show=['horizontal', 'vertical', 'finetune'],
                     zones_to_show=['Safe', 'Dominant', 'Fine-tuning Plateau'],
):
    real_xs, safes, dominants, ft_deltas = get_zones(iters, name, is_iterative, just_get_data=True)
    cmap_lim = 0.005
    norm = MidpointNormalize(midpoint=0, vmin=-cmap_lim, vmax=cmap_lim)
    cmap = matplotlib.cm.get_cmap('bwr')

    for (zone, ztype) in ([
            (safes, 'Safe'),
            (dominants, 'Dominant'),
            (ft_deltas, 'Fine-tuning Plateau'),
    ]):
        if ztype.lower() not in [z.lower() for z in zones_to_show]:
            continue
        fig, ax = plt.subplots()
        plt.title('{} {} Points, {} pruning'.format(
            netname(name),
            ztype,
            'Iterative' if is_iterative else 'One-shot',
        ))

        if zone is not ft_deltas:
            zone = [z for i, z in enumerate(zone) if 0 <= real_xs[i] <= 1]
        C = np.empty((len(zone), len(zone[0][-2])))

        delta = real_xs[4] - real_xs[3]

        for i, z in enumerate(zone):
            for j, diff in enumerate(z[-4]):
                C[i, j] = diff

        all_xs = np.array(dominants[0][-2])

        plt.imshow(C[::-1, :], cmap=cmap, aspect='auto')
        ax.invert_yaxis()
        format_axes(ax)

        class FooSparsityFormatter(ticker.PercentFormatter):
            def __init__(self, *args, flip=True, **kwargs):
                super().__init__(*args, **kwargs)
                self.flip = flip

            def __call__(self, x, i=None):
                zero = np.log(max(all_xs))/np.log(.8)
                x = 0.8 ** (x + zero)
                if self.flip:
                    x = 1 - x
                return super().__call__(x, i)

        class RetrainFormatter(ticker.PercentFormatter):
            def __init__(self, zone, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.zone = zone

            def __call__(self, x, i=None):
                x = np.interp(
                    len(real_xs) - x - 1,
                    np.arange(len(real_xs)),
                    2 - real_xs
                )
                if self.zone is ft_deltas:
                    x = 2 - x
                else:
                    x = x
                return super().__call__(x, i)

        ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, len(zone[0][-2]), 10)))
        ax.xaxis.set_major_formatter(FooSparsityFormatter(1, None if 'resnet' in name else 1, flip=True))
        ax.xaxis.set_minor_locator(ticker.NullLocator())

        ax.set_xlabel('Sparsity')
        if zone is ft_deltas:
            ax.set_ylabel('Re-training Time')
        else:
            ax.set_ylabel('Rewind Point')


        ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, len(zone), 11)))
        ax.yaxis.set_major_formatter(RetrainFormatter(zone, 1))
        ax.yaxis.set_minor_locator(ticker.NullLocator())

        plt.clim(-cmap_lim, cmap_lim)
        plt.tight_layout()

        def format_coord(x,y):
            return "text_string_made_from({:.2f},{:.2f})".format(x,y)
        ax.format_coord = format_coord

        if savename:
            os.makedirs(os.path.join(_DIRNAME, 'figures', savename), exist_ok=True)
            plt.savefig(os.path.join(_DIRNAME, 'figures', savename, '{}.pdf'.format(ztype.replace(' ', ''))), dpi=FIG_DPI)


    for orientation in bars_to_show:
        fig, ax = plt.subplots(figsize=(6, 1.15) if orientation in ('horizontal', 'finetune') else (1.85, 4))
        cbar = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, orientation={'finetune': 'horizontal'}.get(orientation, orientation))
        if orientation == 'finetune':
            cbar.set_label('$\Delta$ Accuracy between Fine-tuning and Plateau')
        else:
            cbar.set_label('$\Delta$ Accuracy between Rewinding and Fine-tuning')


        cbar.set_ticks(np.linspace(0, 1, 3))
        ticks = [t * (cmap_lim * 2) - cmap_lim for t in cbar.get_ticks()]

        if orientation == 'finetune':
            ticks = [-cmap_lim, -cmap_lim/2, 0]
            cbar.set_clim(0, 2)

        cbar.set_ticklabels([
            r'$\leq {:+.2%}$'.format(ticks[0]).replace(r'%', r'\%')
        ] + [
            r'${:+.2%}$'.format(x).replace(r'%', r'\%')
            for x in list(ticks)[1:-1]
        ] + [
            r'$\geq {:+.2%}$'.format(ticks[-1]).replace(r'%', r'\%')
        ])
        fig.tight_layout()

        if savename:
            os.makedirs(os.path.join(_DIRNAME, 'figures', savename), exist_ok=True)
            plt.savefig(os.path.join(_DIRNAME, 'figures', savename, 'bar_{}.pdf'.format(orientation)), dpi=FIG_DPI)

    if show:
        plt.show()
    else:
        plt.close('all')


def lth_plot_iters(iters, name, is_iterative, show=True, savename=None, plot_zones=False):
    if show:
        plt.ion()
    else:
        plt.ioff()

    lines = collections.defaultdict(list)

    all_xs = set()

    for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):
        if it == 'iter_0':
            continue

        if name == 'resnet20' and is_iterative and int(it.split('_')[1]) > 10:
            continue

        if is_iterative:
            density = 0.8 ** int(it.split('_')[1])
        else:
            density = float(it.split('_')[1]) / 100

        if 'vgg' in name and density > 0.2:
            continue

        for (expt, points) in sorted(expts.items()):
            if 'resnet' in name and is_iterative == False and 'prune' not in it:
                continue

            if plot_zones and 'reinit' in expt:
                continue

            xs = []
            y_means = []
            y_errs_lower = []
            y_errs_upper = []

            for (point, delta_dict) in sorted(points.items()):
                deltas = list(delta_dict.values())
                xs.append(point)
                if RANGE_METHOD == MEDIAN_MAX:
                    y_means.append(np.median(deltas))
                    y_errs_lower.append(y_means[-1] - np.min(deltas))
                    y_errs_upper.append(np.max(deltas) - y_means[-1])
                elif RANGE_METHOD == MEAN_STD:
                    y_means.append(np.mean(deltas))
                    y_errs_lower.append(np.std(deltas) if len(deltas) > 1 else 0)
                    y_errs_upper.append(np.std(deltas) if len(deltas) > 1 else 0)
                else:
                    raise ValueError(RANGE_METHOD)

            idx = np.argmax(y_means)

            sparsity = 1 - density
            all_xs.add(density)
            if y_means[idx] < -0.1:
                continue

            lines[expt].append((density, xs[idx], y_means[idx], y_errs_lower[idx], y_errs_upper[idx]))

    all_xs = sorted(set(x[0] for line in lines.values() for x in line))

    fig, lax = plt.subplots()
    for expt in sorted(lines.keys(), key=legend_sort_key):
        plt.errorbar([x[0] for x in lines[expt]], [x[2] for x in lines[expt]],
                     [[x[3] for x in lines[expt]], [x[4] for x in lines[expt]]],
                     fmt=fmtof(expt), label='{} {}'.format(labof(expt), _FLAVOR),
                     color=get_color(expt),
                     capsize=5,
        )



    if is_iterative:
        plt.title('{name} best $\Delta$ accuracy across sparsities, iterative pruning'.format(
            name=netname(name),
        ))
    else:
        plt.title('{name} best $\Delta$ accuracy across sparsities, one-shot pruning'.format(
            name=netname(name),
        ))

    lax.set_ylabel('$\Delta$ accuracy')
    lax.legend(loc='lower left')

    lax.plot([0, max(all_xs)], [0, 0], '--', color=(0,0,0,0.3))
    lax.set_ylim(-0.03, 0.01)
    lax.set_xlabel('Sparsity (log scale)')
    lax.set_xscale('log')
    lax.invert_xaxis()
    xs = np.power(10, np.linspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 10))
    lax.xaxis.set_major_locator(ticker.FixedLocator(xs))
    lax.xaxis.set_major_formatter(SparsityFormatter(1))
    lax.xaxis.set_minor_locator(ticker.NullLocator())
    plt.tight_layout()

    format_axes(lax, plot_zones)

    vals = lax.get_yticks()
    lax.set_yticklabels(['{:+,.2%}'.format(x).replace('%', r'\%') for x in vals])

    if savename:
        os.makedirs(os.path.join(_DIRNAME, 'figures', savename), exist_ok=True)
        plt.savefig(os.path.join(_DIRNAME, 'figures', savename, 'lth.pdf'), dpi=FIG_DPI)

    if not show:
        plt.close('all')

    if show:
        plt.show()



def plot(net, style, filters, *args, **kwargs):
    print('Regular {} {}'.format(net, style))
    latexify()
    iters = globals()['{}_{}_deltas'.format(net, style)]()

    if filters and not callable(filters[0]):
        filters = globals()[filters[0]][filters[1]][1]

    for filt in filters:
        filt(iters)

    plot_iters(iters, net, style=='iterative', *args, **kwargs)

def lth_plot(net, style, filters, *args, do_latexify=True, **kwargs):
    print('LTH {} {}'.format(net, style))
    if do_latexify:
        latexify()
    iters = globals()['{}_{}_deltas'.format(net, style)]()
    for filt in filters:
        filt(iters)
    lth_plot_iters(iters, net, style=='iterative', *args, **kwargs)

def lth_plot_ungenerous(net, style, filters, *args, **kwargs):
    print('LTH {} {}'.format(net, style))
    latexify()
    iters = globals()['{}_{}_deltas'.format(net, style)]()
    for filt in filters:
        filt(iters)
    lth_plot_iters_ungenerous(iters, net, style=='iterative', *args, **kwargs)

def plot_zones(net, style, *args, show=True, do_latexify=True, **kwargs):
    if show:
        plt.ion()
    else:
        plt.ioff()

    print('Zones: {} {}'.format(net, style))
    if do_latexify:
        latexify()

    iters = globals()['{}_{}_deltas'.format(net, style)]()
    for filt in [remove_experiments('reinit'), remove_points(0)]:
        filt(iters)
    get_zones(iters, net, style=='iterative', *args, **kwargs)

    if show:
        plt.show()
    else:
        plt.close('all')

def plot_implots(net, style, *args, show=True, do_latexify=True, **kwargs):
    if show:
        plt.ion()
    else:
        plt.ioff()

    print('Implots: {} {}'.format(net, style))
    if do_latexify:
        latexify()

    iters = globals()['{}_{}_deltas'.format(net, style)]()
    for filt in [remove_experiments('reinit'), remove_points(0)]:
        filt(iters)
    usability_implot(iters, net, style=='iterative', *args, show=show, **kwargs)

def remove_points(*points, experiments=None):
    if isinstance(experiments, str):
        experiments = [experiments]
    def remover(d):
        for it in d:
            for expt in d[it]:
                if experiments is not None and not any(e in expt for e in experiments):
                    continue
                for point in points:
                    if point in d[it][expt]:
                        del d[it][expt][point]
    return remover

def get_zones(iters, name, is_iterative,
              flip=False,
              just_get_data=False,
              gen_individual_plots=False,
              savename=None,
):
    fullname = '{} {}'.format(netname(name), 'Iterative' if is_iterative else 'Oneshot')

    lines = collections.defaultdict(list)

    if is_iterative:
        lname = 'lottery'
        fname = 'finetune'
    else:
        lname = 'oneshot_lottery'
        fname = 'oneshot_finetune'

    if flip:
        fname, lname = lname, fname

    all_xs = set()
    all_expts = set()

    if 'resnet50' in name:
        max_x = 90
    else:
        max_x = 182

    for (it, expts) in sorted(iters.items(), key=lambda x: float(x[0].split('_')[-1])):
        if it == 'iter_0':
            continue

        if name == 'resnet20' and is_iterative and int(it.split('_')[1]) > 10:
            continue

        if is_iterative:
            density = 0.8 ** int(it.split('_')[1])
        else:
            density = float(it.split('_')[1]) / 100

        if 'vgg' in name and density > 0.2:
            continue

        for (expt, points) in sorted(expts.items()):
            if 'resnet' in name and is_iterative == False and 'prune' not in it:
                continue

            xs = []
            y_means = []
            y_errs_lower = []
            y_errs_upper = []

            for (point, delta_dict) in sorted(points.items()):
                deltas = list(delta_dict.values())
                xs.append(point)

                if RANGE_METHOD == MEDIAN_MAX:
                    y_means.append(np.median(deltas))
                    y_errs_lower.append(y_means[-1] - np.min(deltas))
                    y_errs_upper.append(np.max(deltas) - y_means[-1])
                elif RANGE_METHOD == MEAN_STD:
                    y_means.append(np.mean(deltas))
                    y_errs_lower.append(np.std(deltas) if len(deltas) > 1 else 0)
                    y_errs_upper.append(np.std(deltas) if len(deltas) > 1 else 0)
                else:
                    raise ValueError(RANGE_METHOD)

            all_expts.add(expt)

            for plotter in xs:
                plotter /= max_x

                if name == 'resnet50':
                    trep = 90
                else:
                    trep = 182

                if lname in expt:
                    idx = max(i for i in range(len(xs)) if xs[i] <= trep*plotter)
                else:
                    idx = max((y_means[i], i) for i in range(len(xs)) if xs[i] <= trep*plotter)[1]

                sparsity = 1 - density
                all_xs.add(plotter)

                lines[(expt, plotter)].append((density, xs[idx], y_means[idx], y_errs_lower[idx], y_errs_upper[idx]))


    if gen_individual_plots:
        for expt, x in list(lines.keys()):
            if (lname, x) not in lines and (expt, x) != (fname, 2):
                del lines[(expt, x)]

        all_xs = sorted(set(line[0] for line in lines[(fname, 2)]))

        def fmt_plot(ax, title_plotter):
            ax.plot([0, max(all_xs)], [0, 0], '--', color=(0,0,0,0.3))
            ax.set_ylim(-0.03, 0.01)
            ax.set_xlabel('Sparsity (log scale)')
            ax.set_ylabel('$\Delta$ accuracy')
            ax.legend(loc='lower left')
            ax.set_xscale('log')
            ax.invert_xaxis()
            xs = np.power(10, np.linspace(np.log10(min(all_xs)), np.log10(max(all_xs)), 10))
            ax.xaxis.set_major_locator(ticker.FixedLocator(xs))
            ax.xaxis.set_major_formatter(SparsityFormatter(1))
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            plt.tight_layout()
            format_axes(ax)
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:+,.2%}'.format(x).replace('%', r'\%') for x in vals])

            if savename:
                os.makedirs(os.path.join(_DIRNAME, 'figures', savename), exist_ok=True)
                plt.savefig(os.path.join(_DIRNAME, 'figures', savename, '{}.pdf'.format(title_plotter)), dpi=FIG_DPI)


        ded = True
        for key in sorted(lines.keys(), key=lambda x: (x[1], legend_sort_key(x[0]), x[1])):
            if key == (fname, 2):
                continue

            expt, plotter = key
            if 'lottery' in expt:
                if not ded:
                    fmt_plot(lax, title_plotter)
                    plt.legend()
                else:
                    ded = False
                fig, lax = plt.subplots()
                plt.title('{}: Re-train for {:.0%}'.format(fullname, plotter).replace('%', r'\%'))
                maxx = max(x[0] for x in lines[key])
                minx = min(x[0] for x in lines[key])
                plt.plot([minx, maxx], [0, 0], '--', color=(0,0,0,0.3))
                plt.xlim(minx, maxx)
                plt.ylim(-0.03, 0.01)
                title_plotter = '{:.2}'.format(plotter)


                m_key = fname, 2
                plt.errorbar([x[0] for x in lines[m_key]], [x[2] for x in lines[m_key]],
                             [[x[3] for x in lines[m_key]], [x[4] for x in lines[m_key]]],
                             fmt=fmtof(fname), label='{} {}'.format(labof(fname), 'Best'),
                             # color=get_color(expt),
                             capsize=5,
                )

            if expt == lname:
                label = 'Rewind to {:.0%}, replay {:.0%}'.format(
                    1 - plotter, plotter,
                ).replace('%', r'\%')
            else:
                label = '{} at most {:.0%}'.format(
                             labof(expt),
                             plotter,
                         ).replace('%', r'\%')

            plt.errorbar([x[0] for x in lines[key]], [x[2] for x in lines[key]],
                         [[x[3] for x in lines[key]], [x[4] for x in lines[key]]],
                         fmt=fmtof(expt), label=label,
                         # color=get_color(expt),
                         capsize=5,
            )

        fmt_plot(lax, title_plotter)
        # plt.show()
        return


    safes = []
    dominants = []
    ft_deltas = []
    real_xs = []

    for plotter in sorted(list(all_xs)):
        m = lambda s: set(map(operator.itemgetter(0), s))


        if plotter > 1:
            sets = [
                m(lines[(fname, plotter)]),
                m(lines[(fname, 2)])
            ]
        else:
            sets = [
                m(lines[(lname, plotter)]),
                m(lines[(fname, plotter)]),
                m(lines[(fname, 2)]),
            ]

        intersection_densities = set.intersection(*sets)

        if 'vgg16' in name and is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.005 < x < 0.168}
        elif 'vgg16' in name and not is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.005 < x < 0.168}
        elif 'resnet20' in name and is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.1 < x < 0.9}
        elif 'resnet20' in name and not is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.1 < x < 0.9}
        elif 'resnet50' in name and is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.1 < x < 0.9}
        elif 'resnet50' in name and not is_iterative:
            intersection_densities = {x for x in intersection_densities if 0.1 < x < 0.9}

        m_d = []
        s_d = []
        i_d = []


        for d in intersection_densities:
            if plotter <= 1:
                next_m = next(x for x in lines[(lname, plotter)] if x[0] == d)
                m_d.append(next_m)

            next_s = next(x for x in lines[(fname, plotter)] if x[0] == d)
            next_i = next(x for x in lines[(fname, 2)] if x[0] == d)

            s_d.append(next_s)
            i_d.append(next_i)

        real_xs.append(plotter)

        # (density, xs[idx], y_means[idx], y_errs_lower[idx], y_errs_upper[idx])
        def highs(xs):
            return np.array([x[2] + x[4] for x in xs])
        def lows(xs):
            return np.array([x[2] - x[3] for x in xs])
        def mids(xs):
            return np.array([x[2] for x in xs])

        def foo(source, baseline):
            source = sorted(source)[::-1]
            baseline = sorted(baseline)[::-1]
            median = np.median([source[i][2] - baseline[i][2]  for i in range(len(source))])
            mean = np.mean([source[i][2] - baseline[i][2]  for i in range(len(source))])
            ranges_intersect = (highs(source) > lows(baseline)) & (highs(baseline) > lows(source))
            source_range_above_baseline_mid = highs(source) >= mids(baseline)
            baseline_low_above_source_mid = lows(baseline) <= mids(source)
            median_above = mids(source) >= mids(baseline)
            where_median_above = max((source[i][0] for i in range(len(source)) if median_above[i:].all()),
                                     default=-1)

            densities = [s[0] for s in source]

            return (median, mean, ranges_intersect.sum()/len(source), source_range_above_baseline_mid.sum()/len(source), baseline_low_above_source_mid.sum()/len(source), mids(source) - mids(baseline), median_above, densities, where_median_above)


        if plotter <= 1:
            safes.append(foo(m_d, s_d))
            dominants.append(foo(m_d, i_d))
        ft_deltas.append(foo(s_d, i_d))

    real_xs = np.array(real_xs)

    if just_get_data:
        return real_xs, safes, dominants, ft_deltas

    m = {
        'resnet20': 2,
        'resnet50': 0,
        'vgg16': 1,
    }
    idx = m[name] * 2 + int(bool(is_iterative))
    m_bak = {
        i: '{} {}'.format(next(netname(k) for (k, v) in m.items() if v == i//2), 'Iterative' if i%2==1 else 'Oneshot')
        for i in range(6)
    }


    def filtit(idxs):
        # return idxs
        subseqs = []
        prev = None
        subseq = []
        for i in idxs:
            if prev is None or prev == i - 1:
                subseq.append(i)
                prev = i
            else:
                if subseq:
                    subseqs.append(subseq)
                subseq = [i]
                prev = i
        if subseq:
            subseqs.append(subseq)

        return max((len(subseq), random.random(), subseq) for subseq in subseqs)[2]

    def getit(z):
        return filtit([i for i in range(len(real_xs)) if z[i][0] > 0 and z[i][3] >= 1 and z[i][4] >= 1
        ])
    safe_idxs = getit(safes)
    dominant_idxs = getit(dominants)

    safe_xs = [real_xs[i] for i in range(len(real_xs)) if safes[i][-1] != -1]
    dominant_xs = [real_xs[i] for i in range(len(real_xs)) if dominants[i][-1] != -1]

    plt.figure()
    plt.title('Safe and Dominant Zones: {}'.format(fullname))
    plt.plot(safe_xs, [s[-1] for s in safes if s[-1] != -1], 'o--', label='Safe Zone')
    plt.plot(dominant_xs, [s[-1] for s in dominants if s[-1] != -1], 'o--', label='Dominant Zone')
    plt.yscale('log')
    plt.gca().set_yticks([])
    plt.xlim(0, 1)
    plt.xlabel(r'Re-training Percent')
    plt.ylabel('Requisite Sparsity')

    plt.gca().xaxis.set_major_formatter(SparsityFormatter(1, flip=False))
    plt.gca().yaxis.set_major_locator(ticker.FixedLocator([0.8**x for x in range(1, 30)]))
    plt.gca().yaxis.set_major_formatter(SparsityFormatter(1, flip=True))
    plt.gca().yaxis.set_minor_locator(ticker.NullLocator())
    plt.legend()
    plt.tight_layout()
    return


    # plt.title('Safe and Dominant Zones')
    # plt.plot(real_xs[safe_idxs], [idx for _ in range(len(safe_idxs))], 'x:', color='k', label='Safe Zone')
    # plt.plot(real_xs[dominant_idxs], [idx for _ in range(len(dominant_idxs))], 'o-', color='k', label='Dominant Zone')

    # if m[name] * 2 + int(bool(is_iterative)) == 4:
    #     plt.legend()

    # plt.tight_layout()
    # plt.xlim(0, 1)
    # plt.gca().set_yticks(list(range(6)))
    # plt.gca().set_yticklabels(list(map(m_bak.get, range(6))))
    # plt.gca().set_xticks(np.linspace(0, 1, 11))
    # plt.gca().set_xticklabels(['{:.0%}'.format(x).replace('%', r'\%') for x in np.linspace(0, 1, 11)])
    # plt.grid(True, axis='x')
    # plt.grid(True, which='minor', linestyle='--', linewidth=1)

    # plt.tight_layout()

def remove_other_points(*points, experiments=None):
    if isinstance(experiments, str):
        experiments = [experiments]
    def remover(d):
        for it in d:
            for expt in d[it]:
                if experiments is not None and not any(e in expt for e in experiments):
                    continue
                for point in list(d[it][expt].keys()):
                    if point not in points:
                        del d[it][expt][point]
    return remover

def remove_experiments(*experiments):
    def remover(d):
        for it in d:
            for expt in list(d[it].keys()):
                if any(e in expt for e in experiments):
                    del d[it][expt]
    return remover


def netname(net):
    return {
        'resnet20': 'ResNet-20',
        'resnet50': 'ResNet-50',
        'vgg16': 'VGG-16',
        'vgg19': 'VGG-19',
    }.get(net, net)


def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        import math
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 15,
        "font.size": 15,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (7, 3.5),
        "figure.dpi": 100,
        "legend.loc": 'best',
    }

    matplotlib.rcParams.update(nice_fonts)


def format_axes(ax, twinx=False):
    SPINE_COLOR = 'gray'

    if twinx:
        visible_spines = ['bottom', 'right']
        invisible_spines = ['top', 'left']
    else:
        visible_spines = ['bottom', 'left']
        invisible_spines = ['top', 'right']

    for spine in invisible_spines:
        ax.spines[spine].set_visible(False)

    for spine in visible_spines:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')

    if twinx:
        ax.yaxis.set_ticks_position('right')
    else:
        ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

def print_table(output=None):

    if output is None:
        f = sys.stdout
    else:
        os.makedirs(os.path.join(_DIRNAME, 'figures'), exist_ok=True)
        f = open(os.path.join(_DIRNAME, 'figures', output), 'w')

    with contextlib.redirect_stdout(f):
        rowidx = 0

        print(r'\begin{tabular}{ccccc}')
        print(r'\toprule')
        print(r'\textbf{Network} & \textbf{Pruning Method} & \textbf{Finetune Sparsity} & \textbf{Rewinding Sparsity} & \textbf{Ratio} \\ \midrule')
        fst = True

        for net in ['resnet20', 'vgg16', 'resnet50']:
            for pmethod in ['oneshot', 'iterative']:
                methods = ['finetune', 'lottery']
                res = {}
                for tmethod in methods:
                    iters = globals()['{}_{}_deltas'.format(net, pmethod)]()
                    res[tmethod] = collections.defaultdict(lambda: (float('-inf'), float('nan')))

                    for k in iters:
                        m_density = None
                        if k.startswith('iter_'):
                            m_density = (0.8 ** int(k[5:]))
                        elif k.startswith('prune_'):
                            m_density = float(k[6:]) / 100
                        else:
                            raise ValueError(k)

                        m_sparsity = 1 - m_density

                        for method in iters[k]:
                            if tmethod not in method:
                                continue
                            for m_rt_epochs in iters[k][method]:
                                if RANGE_METHOD == MEDIAN_MAX:
                                    m_acc = np.max(list(iters[k][method][m_rt_epochs].values()))
                                    m_err_low = m_acc - np.min(list(iters[k][method][m_rt_epochs].values()))
                                    m_err_high = np.max(list(iters[k][method][m_rt_epochs].values())) - m_acc
                                elif RANGE_METHOD == MEAN_STD:
                                    m_acc = np.mean(list(iters[k][method][m_rt_epochs].values()))
                                    m_err_low = np.std(list(iters[k][method][m_rt_epochs].values()))
                                    m_err_high = np.std(list(iters[k][method][m_rt_epochs].values()))
                                else:
                                    raise ValueError(RANGE_METHOD)
                                res[tmethod][m_sparsity] = max(res[tmethod][m_sparsity], (m_acc, m_err_low, m_err_high, m_rt_epochs))

                ft_sparsity = max((sparsity for (sparsity, (acc, low, high, rte)) in res['finetune'].items() if acc + high > 0), default=None)
                lot_sparsity = max((sparsity for (sparsity, (acc, low, high, rte)) in res['lottery'].items() if acc + high > 0), default=None)

                print(r'{network} & {pmethod} & {{ {ft_bold} {ft_sparsity:.2%} }} & {{ {lot_bold} {lot_sparsity:.2%} }} & ${ratio:.2f} \times$ \\'.format(
                    network=netname(net),
                    pmethod=pmethod.replace('oneshot', 'one-shot').capitalize(),
                    ft_bold=r'\bf' if ft_sparsity > lot_sparsity else '',
                    ft_sparsity=ft_sparsity,
                    lot_bold=r'\bf' if lot_sparsity > ft_sparsity else '',
                    lot_sparsity=lot_sparsity,
                    ratio=(1-ft_sparsity)/(1-lot_sparsity)
                ).replace('%', r'\%'))

        print(r'\bottomrule')
        print(r'\end{tabular}')

def get_surfaces(iters):
    surfaces = collections.defaultdict(list)
    for it in iters:
        if 'iter_' in it:
            density = 0.8 ** int(it.split('_')[1])
        else:
            density = float(it.split('_')[1]) / 100

        for expt in iters[it]:
            for point in iters[it][expt]:
                acc = np.mean(list(iters[it][expt][point].values()))
                surfaces[expt].append((density, point, acc))

    return surfaces


cifar_configs = [
    ('main', [remove_points(0, 192, 212, 232, 252, 254, 272, 292, 312, 314, 332, 352, 364), remove_experiments('reinit')]),
    # ('ftext', [remove_points(0, 182, 254, 314), remove_experiments('reinit')]),
    # ('reext', [remove_points(0, 182, 254, 314), remove_experiments('finetune')]),
]

imagenet_configs = [
    ('main', [remove_points(0, 99, 108, 117, 126, 135, 144, 153, 162, 171, 180), remove_experiments('reinit')]),
    # ('ftext', [remove_points(0), remove_points(90, experiments='lottery'), remove_experiments('reinit')]),
    # ('reext', [remove_points(0), remove_points(90, experiments='lottery'), remove_experiments('finetune')]),
]

def runner(func, args, kwargs):
    mpldir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, mpldir)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(mpldir, 0o777 & ~umask)
    os.environ['HOME'] = mpldir
    os.environ['MPLCONFIGDIR'] = mpldir
    import matplotlib
    class TexManager(matplotlib.texmanager.TexManager):
        texcache = os.path.join(mpldir, 'tex.cache')
    matplotlib.texmanager.TexManager = TexManager
    matplotlib.rcParams['ps.useafm'] = True
    matplotlib.rcParams['pdf.use14corefonts'] = True
    matplotlib.rcParams['text.usetex'] = True

    func(*args, **kwargs)

def gen_all_the_plots():
    multiprocessing.set_start_method('spawn')
    coms = []
    coms.append((print_table, (), {'output': 'table.tex'}))

    for net in ['resnet50', 'resnet20', 'vgg16']:
        for i in ['oneshot', 'iterative']:
            configs = imagenet_configs if net == 'resnet50' else cifar_configs
            for idx, (name, filters) in enumerate(configs):
                savename = '{}_{}_{}'.format(
                    net, i, name
                )
                coms.append((plot, (net, i, ('{}_configs'.format('imagenet' if net == 'resnet50' else 'cifar'), idx)), {
                    'show': False,
                    'savename': savename}))
            coms.append((lth_plot, (net, i, []), {
                'show': False,
                'savename': '{}_{}'.format(net, i)}))
            coms.append((plot_implots, (net, i), {
                'show': False,
                'savename': '{}_{}_zone'.format(net, i)}))
            coms.append((plot_zones, (net, i), {
                'gen_individual_plots': True,
                'show': False,
                'savename': '{}_{}_zone_raw'.format(net, i)}))

    p = multiprocessing.Pool(min(len(coms), 32))
    res = []

    for (f, args, kwargs) in coms:
        res.append(p.apply_async(runner, (f, args, kwargs)))

    for (com, r) in zip(coms, res):
        print('Waiting {}'.format(com))
        r.get()

if __name__ == '__main__':
    gen_all_the_plots()
