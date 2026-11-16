"""Draw the two-population repeated-CV comparison from its source tables."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

INK, BLUE, PINK, GRAY = '#28343D', '#8BAFCB', '#C8739D', '#A8B1B8'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.linewidth': .75,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK})

def panel(ax, frame, name, n, letter):
    gw = frame['Genome_wide_r'].to_numpy(float)
    qt = frame['QTL_informed_r'].to_numpy(float)
    for x, values, side, color in [(0.8, gw, -1, BLUE), (2.1, qt, 1, PINK)]:
        y = np.linspace(values.min() - .025, values.max() + .025, 240)
        density = gaussian_kde(values, bw_method='scott')(y)
        width = .39 * density / density.max()
        ax.fill_betweenx(y, x, x + side * width, color=color, alpha=.45, lw=.8, zorder=1)
        for value in np.quantile(values, [.25, .5, .75]):
            ax.plot([x, x + side * float(np.interp(value, y, width))], [value, value],
                    color='#73818A', ls='--', lw=.6, alpha=.55, zorder=2)
    for a, b in zip(gw, qt):
        ax.plot([1.08, 1.82], [a, b], color=GRAY, lw=.65, alpha=.82, zorder=2)
    for x, values, color, boxcolor, side in [(0.8, gw, BLUE, '#7FA4C1', 1), (2.1, qt, PINK, '#BC638E', -1)]:
        bx = x + side * .12
        ax.boxplot(values, positions=[bx], widths=.18, patch_artist=True, showfliers=False, whis=(0, 100),
                   boxprops={'facecolor': boxcolor, 'edgecolor': boxcolor, 'lw': .7},
                   whiskerprops={'color': '#4B5660', 'lw': .8}, capprops={'color': '#4B5660', 'lw': .8},
                   medianprops={'color': '#56636C', 'lw': .7}, zorder=3)
        ax.scatter(np.full(len(values), x + side * .28), values, s=12, color=color, edgecolors='none', zorder=4)
        ax.scatter([bx], [values.mean()], marker='D', s=23, facecolor='white', edgecolor=INK, lw=.75, zorder=6)
    ax.set(xlim=(.25, 2.65), ylim=(.275, .58), yticks=np.arange(.28, .561, .04), ylabel='Pearson r')
    ax.set_xticks([.8, 2.1], ['Genome-wide', 'QTL-informed'], fontsize=8)
    ax.tick_params(axis='x', length=0, pad=5)
    ax.tick_params(axis='y', length=3, labelsize=7.5, labelleft=True)
    ax.yaxis.labelpad = 4
    ax.set_title(f'{name} population (n = {n})', fontsize=9, pad=10)
    ax.text(-.095, 1.085, letter, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#AAB3B8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    primary = pd.read_csv(args.data / 'Figure3_development_source_data.csv').sort_values('repeat')
    validation = pd.read_csv(args.data / 'paired_repeat_results.csv').sort_values('repeat').rename(columns={
        'Genome_wide_pearson_r': 'Genome_wide_r', 'QTL_informed_pearson_r': 'QTL_informed_r'})
    for frame in (primary, validation):
        if frame['repeat'].tolist() != list(range(1, 21)) or not np.isfinite(frame[['Genome_wide_r', 'QTL_informed_r']].to_numpy()).all():
            raise ValueError('Expected 20 complete, finite repeat-level results.')
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 98 / 25.4), sharey=True)
    fig.subplots_adjust(left=.085, right=.978, bottom=.11, top=.85, wspace=.32)
    panel(axes[0], primary, 'Primary', 241, 'a')
    panel(axes[1], validation, 'Validation', 151, 'b')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(args.output / f'Figure4.{ext}', dpi=300)
    plt.close(fig)

if __name__ == '__main__':
    main()
