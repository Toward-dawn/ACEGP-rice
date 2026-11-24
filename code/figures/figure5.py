"""Draw Figure 5 from four supplied source tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import ConnectionPatch
BLUE = '#4F8FB8'
DARK = '#2E648A'
FILL = '#BFD9E8'
MAGENTA = '#A94368'
INK = '#20242A'

def kde_counts(values, x, bin_width):
    n = len(values)
    bandwidth = max(1.06 * np.std(values, ddof=1) * n ** (-1 / 5), bin_width * 0.55, 1e-06)
    assert n == 100 and np.isfinite(values).all() and (bandwidth > 0)
    z = (x[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return (density * n * bin_width, float(bandwidth))

def format_inset(ax):
    ax.set_facecolor('white')
    ax.tick_params(labelsize=5.0, pad=1.2, length=1.7, width=0.4)
    ax.grid(axis='y', color='#EEF2F5', lw=0.35)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.45)
        spine.set_color('#9AA5AE')
    ax.text(0.03, 0.95, '95% percentile interval', transform=ax.transAxes, ha='left', va='top', fontsize=5.0, color=DARK, zorder=8, bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.4, 'alpha': 0.9})

def callout(ax, label, observed, is_gain, yfrac):
    position = 0.89 if observed else 0.44
    x = ax.get_xlim()[0] + position * (ax.get_xlim()[1] - ax.get_xlim()[0])
    y = ax.get_ylim()[0] + yfrac * (ax.get_ylim()[1] - ax.get_ylim()[0])
    target = label[1]
    color = MAGENTA if observed else DARK
    value = f'Δr = {target:+.4f}' if is_gain else f'r = {target:.4f}'
    return ax.annotate(label[0] + '\n' + value, xy=(target + (0.003 if not observed else -0.003), y + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])), xytext=(x, y), ha='right' if observed else 'left', va='center', fontsize=6.3, color=color, bbox={'boxstyle': 'round,pad=0.24', 'fc': 'white', 'ec': MAGENTA if observed else BLUE, 'lw': 0.45}, arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 0.55, 'mutation_scale': 5.5}, zorder=8)

def connect(ax, inset, lo, hi, y, fig):
    ax.plot([lo, hi], [y, y], color=BLUE, lw=0.5, ls=(0, (3, 2)), alpha=0.7, zorder=1)
    for x, j in [(lo, 0), (hi, 1)]:
        fig.add_artist(ConnectionPatch(xyA=(x, y), coordsA=ax.transData, xyB=(j, 0), coordsB=inset.transAxes, color=BLUE, lw=0.5, ls=(0, (3, 2)), alpha=0.7, zorder=1))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'text.color': INK})
    fig = plt.figure(figsize=(240 / 25.4, 303 / 25.4))
    xs = [0.065, 0.545]
    ys = [0.704, 0.381, 0.058]
    w = 0.42
    h = 0.25
    all_axes = [fig.add_axes([x, y, w, h]) for y in ys for x in xs]
    a, b = all_axes[:2]
    axes = all_axes[2:]
    d = pd.read_csv(args.data / 'Figure4_random100_source_data.csv').sort_values('panel')
    if d['panel'].tolist() != list(range(1, 101)):
        raise ValueError('Fixed-split panel IDs must be unique and cover 1–100.')
    refs = pd.read_csv(args.data / 'Figure4_reference_values.csv').set_index('reference')
    values = d.pearson_r.to_numpy(float)
    gains = d.delta_r_vs_same_GW.to_numpy(float)
    real = refs.loc['QTL-informed', 'pearson_r']
    gw = refs.loc['Genome-wide', 'pearson_r']
    mean = refs.loc['Random-panel mean', 'pearson_r']
    qlo, qhi = np.quantile(values, [0.025, 0.975])
    glo, ghi = np.quantile(gains, [0.025, 0.975])
    span = np.ptp(values)
    bins = np.linspace(values.min() - span * 0.035, values.max() + span * 0.035, 17)
    local_bins = np.linspace(values.min() - span * 0.015, values.max() + span * 0.015, 13)
    counts, _ = np.histogram(values, bins=bins)
    local_counts, _ = np.histogram(values, bins=local_bins)
    ymax = float(4 * np.ceil(max(counts) * 1.38 / 4))
    main_limits = (0.26, 0.5)
    a.set(xlim=main_limits, ylim=(0, ymax), xlabel='Pearson r', ylabel='Number of random panels')
    b.set(xlim=(main_limits[0] - gw, main_limits[1] - gw), ylim=(0, 1.06), xlabel='Prediction gain Δr', ylabel='Empirical cumulative probability')
    for ax in [a, b]:
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#E8EDF1', lw=0.4)
    a.axvspan(qlo, qhi, color=FILL, alpha=0.38, lw=0, zorder=0)
    a.hist(values, bins=bins, color=FILL, edgecolor=BLUE, lw=0.45, alpha=0.86, zorder=2)
    x = np.linspace(bins[0], bins[-1], 500)
    density, bandwidth = kde_counts(values, x, float(np.diff(bins).mean()))
    a.plot(x, density, color=DARK, lw=0.9, zorder=3)
    a.axvline(mean, color=DARK, ls=(0, (5, 3)), lw=0.8, zorder=4)
    a.axvline(real, color=MAGENTA, lw=3.0, zorder=5)
    a.set_xticks([0.28, 0.32, 0.36, 0.4, 0.44, 0.48])
    a.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_yticks(np.arange(0, ymax + 1, 4))
    sg = np.sort(gains)
    ecdf = np.arange(1, 101) / 100
    b.axvspan(glo, ghi, color=FILL, alpha=0.42, lw=0, zorder=0)
    b.step(sg, ecdf, where='post', color=DARK, lw=1.0, zorder=3)
    b.fill_between(sg, ecdf, 0, step='post', color=FILL, alpha=0.34, zorder=1)
    b.axvline(mean - gw, color=DARK, ls=(0, (5, 3)), lw=0.8, zorder=4)
    b.axvline(real - gw, color=MAGENTA, lw=3.0, zorder=5)
    b.set_xticks([-0.05, 0, 0.05, 0.1, 0.15])
    b.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    b.set_yticks([0, 0.25, 0.5, 0.75, 1])
    b.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    geometry = [0.48, 0.4, 0.38, 0.44]
    ia = a.inset_axes(geometry)
    ib = b.inset_axes(geometry)
    ia.axvspan(qlo, qhi, color=FILL, alpha=0.42, lw=0, zorder=0)
    ia.hist(values, bins=local_bins, color=FILL, edgecolor=BLUE, lw=0.35, alpha=0.9, zorder=2)
    xx = np.linspace(local_bins[0], local_bins[-1], 350)
    local_density, local_bandwidth = kde_counts(values, xx, float(np.diff(local_bins).mean()))
    ia.plot(xx, local_density, color=DARK, lw=0.7, zorder=3)
    ia.axvline(mean, color=DARK, ls=(0, (4, 2)), lw=0.6, zorder=4)
    ia.eventplot(values, lineoffsets=-0.42, linelengths=0.24, linewidths=0.3, colors=BLUE, alpha=0.55, zorder=4)
    local_limits = (float(local_bins[0] - span * 0.01), float(local_bins[-1] + span * 0.01))
    ia.set(xlim=local_limits, ylim=(-1.1, max(10, float(local_counts.max()) * 1.35)))
    ia.set_xticks([0.28, 0.32, 0.36])
    ia.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ia.set_yticks([0, 10, 20])
    ib.axvspan(glo, ghi, color=FILL, alpha=0.42, lw=0, zorder=0)
    ib.step(sg, ecdf, where='post', color=DARK, lw=0.7, zorder=3)
    ib.fill_between(sg, ecdf, 0, step='post', color=FILL, alpha=0.34, zorder=1)
    ib.axvline(mean - gw, color=DARK, ls=(0, (4, 2)), lw=0.6, zorder=4)
    ib.eventplot(gains, lineoffsets=-0.035, linelengths=0.055, linewidths=0.3, colors=BLUE, alpha=0.55, zorder=4)
    ib.set(xlim=(local_limits[0] - gw, local_limits[1] - gw), ylim=(-0.08, 1.04))
    ib.set_xticks([-0.04, 0, 0.03])
    ib.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ib.set_yticks([0, 0.5, 1])
    ib.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    for inset in [ia, ib]:
        format_inset(inset)
    connect(a, ia, qlo, qhi, ymax * 0.27, fig)
    connect(b, ib, glo, ghi, 0.32, fig)
    boxes = [callout(a, ('Random-panel mean', mean), False, False, 0.16), callout(a, ('Observed 8-QTL', real), True, False, 0.16), callout(b, ('Random-panel mean', mean - gw), False, True, 0.2 / 1.06), callout(b, ('Observed 8-QTL', real - gw), True, True, 0.2 / 1.06)]
    low = pd.read_csv(args.data / 'TABLE_A_panel_means.csv')
    high = pd.read_csv(args.data / 'review_per_repeat.csv')
    expected_panels = {f'random_{i:03d}' for i in range(1, 101)} | {'real_sparse_QTL'}
    for cohort in ('development', 'validation'):
        group = low[low.cohort.eq(cohort)]
        if len(group) != 101 or set(group.block) != expected_panels or group.block.duplicated().any():
            raise ValueError(f'{cohort}: sparse panels must contain one real panel and random_001–random_100.')
        repeats = high.loc[high.cohort.eq(cohort), 'repeat'].tolist()
        if len(repeats) != 20 or set(repeats) != set(range(1, 21)):
            raise ValueError(f'{cohort}: high-density repeats must be unique and cover 1–20.')
    for ax, cohort, title in zip(axes[:2], ['development', 'validation'], ['Primary population', 'Validation population']):
        t = low[low.cohort.eq(cohort)]
        vals = t[t.block.str.startswith('random_')].r_addon_mean.to_numpy()
        obs = t[t.block.eq('real_sparse_QTL')].r_addon_mean.to_numpy()
        assert len(vals) == 100 and len(obs) == 1
        span = np.ptp(vals)
        bins = np.linspace(vals.min() - span * 0.035, vals.max() + span * 0.035, 17)
        counts, _ = np.histogram(vals, bins=bins)
        bw = max(1.06 * np.std(vals, ddof=1) * len(vals) ** (-0.2), np.diff(bins)[0] * 0.55, 1e-06)
        x = np.linspace(bins[0], bins[-1], 400)
        density = np.exp(-0.5 * ((x[:, None] - vals) / bw) ** 2).sum(axis=1) / (len(vals) * bw * np.sqrt(2 * np.pi)) * len(vals) * np.diff(bins)[0]
        ymax = max(max(counts), max(density)) * 1.35
        ax.hist(vals, bins=bins, color=FILL, edgecolor=BLUE, lw=0.45, alpha=0.86)
        ax.plot(x, density, color=DARK, lw=0.9)
        ax.axvline(vals.mean(), color=DARK, ls=(0, (5, 3)), lw=0.8)
        ax.axvline(obs[0], color=MAGENTA, lw=3.0)
        ax.annotate(f'QTL panel\nr = {obs[0]:.4f}', xy=(obs[0], ymax * 0.62), xytext=(-12, 0), textcoords='offset points', ha='right', va='center', fontsize=8, color=MAGENTA, bbox=dict(boxstyle='round,pad=.24', fc='white', ec=MAGENTA, lw=0.45), arrowprops=dict(arrowstyle='-|>', color=MAGENTA, lw=0.55, mutation_scale=5.5))
        ax.set_ylim(0, ymax)
        ax.set_title(title, fontsize=8, pad=6)
        ax.set_xlabel('Mean Pearson r across 20 repeats')
        ax.set_ylabel('Number of random panels')
        ax.grid(axis='y', color='#E8EDF1', lw=0.4)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.annotate(f'Random-panel mean\nr = {vals.mean():.4f}', xy=(vals.mean(), ymax * 0.62), xytext=(25, 0), textcoords='offset points', ha='left', va='center', fontsize=8, color=DARK, bbox=dict(boxstyle='round,pad=.24', fc='white', ec=BLUE, lw=0.45), arrowprops=dict(arrowstyle='-|>', color=DARK, lw=0.55, mutation_scale=5.5))
    for ax, cohort, title in zip(axes[2:], ['development', 'validation'], ['Primary population', 'Validation population']):
        v = high[high.cohort.eq(cohort)].sort_values('repeat')
        assert len(v) == 20
        for _, row in v.iterrows():
            ax.plot([0, 1], [row.r_genome, row.r_qtl], color='#B8C5CE', lw=0.6, alpha=0.7, zorder=1)
        for j, (col, color) in enumerate([('r_genome', BLUE), ('r_qtl', MAGENTA)]):
            vals = v[col].to_numpy()
            ax.scatter(np.full(20, j), vals, s=44, color=color, alpha=0.58, edgecolor='white', linewidth=0.25, zorder=2)
            mean = vals.mean()
            ax.scatter([j], [mean], marker='D', s=44, color=color, edgecolor='white', linewidth=0.5, zorder=4)
            ax.annotate(f'{mean:.4f}', (j, mean), xytext=(8, 2), textcoords='offset points', color=color, fontsize=8)
        ax.set(xlim=(-0.4, 1.55), xticks=[0, 1], xticklabels=['Genome-wide', 'QTL-informed'], ylabel='Pearson r')
        ax.set_title(title, fontsize=8, pad=6)
        ax.grid(axis='y', color='#E8EDF1', lw=0.4)
        ax.set_axisbelow(True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.15)
        ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#666666', alpha=0.75, markersize=np.sqrt(44), markeredgecolor='white', markeredgewidth=0.25, label='Individual repeat'), Line2D([], [], marker='D', ls='', color='#666666', markersize=np.sqrt(44), markeredgecolor='white', markeredgewidth=0.5, label='Mean')], loc='upper left', bbox_to_anchor=(0.02, 0.93), frameon=False, fontsize=8, borderaxespad=0, handletextpad=0.4, handlelength=0.8, borderpad=0)
    for ax in all_axes:
        ax.title.set_fontsize(8)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.tick_params(labelsize=8)
        for t in ax.texts:
            t.set_fontsize(8)
    for ax in axes[2:]:
        ax.tick_params(axis='x', labelsize=10)
    for ax in all_axes[:4]:
        for line in ax.lines:
            if line.get_color() == MAGENTA and line.get_linewidth() == 3.0:
                line.set_linewidth(4.2)
    for ax in axes:
        ax.set_title(ax.get_title(), fontsize=9, pad=2)
    for ax, box in [(a, boxes[0]), (b, boxes[2])]:
        x, y = box.get_position()
        box.set_position((x - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.09, y))
    for inset in [ia, ib]:
        inset.tick_params(labelsize=6)
        for t in inset.texts:
            t.set_fontsize(6)
    for i, label in enumerate('abcdef'):
        fig.text(xs[i % 2] - (0.052 if i % 2 == 0 else 0.038), ys[i // 2] + h + 0.013, label, fontsize=13, fontweight='bold', va='bottom')
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(args.output / f'Figure5.{ext}', dpi=300)
    plt.close(fig)
if __name__ == '__main__':
    main()
