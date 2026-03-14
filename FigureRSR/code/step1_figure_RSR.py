#!/usr/bin/env python3
"""Scatter subplots of r_eq vs distance for 6 leaf condensation trials."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR    = THIS_DIR.parent / 'raw_data'

# Plot style
MM = 1 / 25.4
TS = 6.5;  LS = 7.5;  PL = 10.0
LW = 0.6

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          TS,
    'axes.linewidth':     LW,
    'xtick.major.width':  LW, 'ytick.major.width': LW,
    'xtick.major.size':   3.0, 'ytick.major.size': 3.0,
    'xtick.direction':    'out', 'ytick.direction': 'out',
    'svg.fonttype':       'none',
})

# Trials
HEALTHY  = ['RSR1', 'RSR2', 'RSR7']
DISEASED = ['RSRDiseased3', 'RSRDiseased5', 'RSRDiseased6']


def main():
    df = pd.read_csv(RAW_DIR / 'droplets_calibrated_mm.csv')
    print(f'Loaded {len(df)} droplets from {df["sample"].nunique()} samples')

    fig, axes = plt.subplots(3, 2, figsize=(100 * MM, 120 * MM),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.20,
                        left=0.14, right=0.96, top=0.92, bottom=0.10)

    # Column headers
    axes[0, 0].set_title('Healthy', fontsize=LS, fontweight='bold', pad=6)
    axes[0, 1].set_title('Diseased', fontsize=LS, fontweight='bold', pad=6)

    for col, trial_list in enumerate([HEALTHY, DISEASED]):
        for row, sample in enumerate(trial_list):
            ax = axes[row, col]
            sub = df[df['sample'] == sample].copy()

            x = sub['dist_mm'].values
            y = sub['r_eq_mm'].values

            # Scatter
            ax.scatter(x, y, s=3, c='#888888', alpha=0.4,
                       edgecolors='none', rasterized=True, zorder=1)

            # Linear regression
            valid = np.isfinite(x) & np.isfinite(y)
            sl, ic, r, p, se = stats.linregress(x[valid], y[valid])
            xfit = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(xfit, ic + sl * xfit, color='#C0392B', lw=1.2, zorder=3)

            # Spines
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)

    # Shared axis labels
    fig.text(0.55, 0.02, 'Distance from boundary (mm)',
             ha='center', fontsize=LS)
    fig.text(0.02, 0.52, r'$r_{\mathrm{eq}}$ (mm)',
             ha='center', va='center', rotation=90, fontsize=LS)

    # Panel label
    axes[0, 0].text(-0.35, 1.15, 'B', transform=axes[0, 0].transAxes,
                    fontsize=PL, fontweight='bold', va='top')

    # Save
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panel_B_RSR_scatter{ext}', **kw)
    plt.close(fig)
    print(f'\nSaved to {OUTPUT_DIR}/panel_B_RSR_scatter.*')


if __name__ == '__main__':
    main()
