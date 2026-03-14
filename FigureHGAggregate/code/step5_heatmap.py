#!/usr/bin/env python3
"""Heatmap of normalised droplet radius R* vs distance for all 30 trials."""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# Paths
THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HG_AGG_DIR    = PROJECT_DIR / 'raw_data' / 'aggregate_edt'
FUNGI_AGG_DIR = PROJECT_DIR.parent / 'FigureFungi' / 'raw_data' / 'aggregate_edt'

# Trial ordering
TRIALS = [
    ('agar.1', 'Agar'),   ('agar.2', 'Agar'),   ('agar.3', 'Agar'),
    ('agar.4', 'Agar'),   ('agar.5', 'Agar'),
    ('1to1.1', '1:1'),    ('1to1.2', '1:1'),    ('1to1.3', '1:1'),
    ('1to1.4', '1:1'),    ('1to1.5', '1:1'),
    ('2to1.1', '2:1'),    ('2to1.2', '2:1'),    ('2to1.3', '2:1'),
    ('2to1.4', '2:1'),    ('2to1.5', '2:1'),
    ('Green.1', 'Green'),  ('Green.2', 'Green'),  ('Green.3', 'Green'),
    ('Green.4', 'Green'),  ('Green.5', 'Green'),
    ('white.1', 'White'),  ('white.2', 'White'),  ('white.3', 'White'),
    ('white.4', 'White'),  ('white.5', 'White'),
    ('black.1', 'Black'),  ('black.2', 'Black'),  ('black.3', 'Black'),
    ('black.4', 'Black'), ('black.5', 'Black'),
]

# Pre-computed raycast delta (µm)
DELTA = {
    'agar.1':  77.8,  'agar.2':  92.4,  'agar.3': 163.5,
    'agar.4': 100.1,  'agar.5': 101.6,
    '1to1.1': 420.5,  '1to1.2': 286.6,  '1to1.3': 420.1,
    '1to1.4': 363.7,  '1to1.5': 462.7,
    '2to1.1': 681.1,  '2to1.2': 803.2,  '2to1.3': 933.4,
    '2to1.4': 872.9,  '2to1.5': 1005.4,
    'Green.1': 279.5,  'Green.2': 316.0,  'Green.3': 297.8,
    'Green.4': 285.6,  'Green.5': 311.7,
    'white.1': 198.5,  'white.2': 126.2,  'white.3': 120.0,
    'white.4': 131.6,  'white.5': 123.0,
    'black.1': 125.4,  'black.2':  98.0,  'black.3': 111.7,
    'black.4': 135.7, 'black.5': 78.3,
}

# Parameters
T_WINDOW      = (14.5, 15.5)
BIN_WIDTH_UM  = 50
MIN_DROPS_BIN = 10
DIST_MAX_UM   = 2500
DIST_GRID     = np.arange(BIN_WIDTH_UM / 2, DIST_MAX_UM + 1, BIN_WIDTH_UM)

# Plot style
MM = 1 / 25.4
TS = 6.5
LS = 8.0
LW = 0.6

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TS,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'svg.fonttype': 'none',
})

GROUP_COLORS = {
    'Agar': '#3A9E6F', '1:1': '#3A6FBF', '2:1': '#C0392B',
    'Green': '#4CAF50', 'White': '#9E9E9E', 'Black': '#212121',
}


def load_and_bin(trial_id):
    """Load per-droplet CSV, bin at BIN_WIDTH_UM, return (bin_centres, mean_R)."""
    for d in [HG_AGG_DIR, FUNGI_AGG_DIR]:
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            break
    else:
        return None, None

    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1])].copy()
    if len(tw) < 30:
        return None, None

    bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um']
    prof = grp.agg(r='mean', n='count').reset_index()
    prof = prof[prof['n'] >= MIN_DROPS_BIN].sort_values('bin')

    return prof['bin'].values, prof['r'].values


def main():
    n_trials = len(TRIALS)
    n_bins = len(DIST_GRID)

    matrix = np.full((n_trials, n_bins), np.nan)
    r_far = np.full(n_trials, np.nan)

    for i, (tid, grp) in enumerate(TRIALS):
        d_vals, r_vals = load_and_bin(tid)
        if d_vals is None:
            print(f'  [SKIP] {tid}')
            continue

        # Far-field reference: mean R in top 25% of distance range
        d_75 = np.percentile(d_vals, 75)
        far_mask = d_vals >= d_75
        if far_mask.sum() >= 2:
            r_far[i] = r_vals[far_mask].mean()
        else:
            r_far[i] = r_vals[-3:].mean()

        # Place into common grid
        for d, r in zip(d_vals, r_vals):
            j = np.argmin(np.abs(DIST_GRID - d))
            if abs(DIST_GRID[j] - d) <= BIN_WIDTH_UM:
                matrix[i, j] = r

        print(f'  {tid:<14s}  R_far={r_far[i]:.1f} µm  '
              f'bins={len(d_vals)}  d={d_vals.min():.0f}-{d_vals.max():.0f}')

    # Normalise each row: R* = R / R_far
    for i in range(n_trials):
        if not np.isnan(r_far[i]) and r_far[i] > 0:
            matrix[i, :] /= r_far[i]

    print(f"\nMatrix: {n_trials} trials x {n_bins} bins")
    print(f"R* range: {np.nanmin(matrix):.2f} - {np.nanmax(matrix):.2f}")
    print(f"NaN fraction: {np.isnan(matrix).sum() / matrix.size:.1%}")

    # Figure
    fig_w = 140 * MM
    fig_h = 145 * MM
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    fig.subplots_adjust(left=0.14, right=0.74, top=0.96, bottom=0.09)

    # Colormap: viridis with NaN as light grey
    cmap = cm.viridis.copy()
    cmap.set_bad(color='#EDEDED')

    # Distance edges for pcolormesh (mm)
    half = BIN_WIDTH_UM / 2
    x_edges = np.concatenate([DIST_GRID - half,
                              [DIST_GRID[-1] + half]]) / 1000.0
    y_edges = np.arange(n_trials + 1) - 0.5

    im = ax.pcolormesh(x_edges, y_edges, matrix,
                       cmap=cmap, vmin=0, vmax=1.15,
                       rasterized=True)

    # Delta markers
    for i, (tid, grp) in enumerate(TRIALS):
        delta_mm = DELTA.get(tid, np.nan) / 1000.0
        if np.isnan(delta_mm):
            continue
        ax.plot([delta_mm, delta_mm], [i - 0.4, i + 0.4],
                color='white', lw=1.8, solid_capstyle='round', zorder=5)
        ax.plot([delta_mm, delta_mm], [i - 0.4, i + 0.4],
                color='red', lw=0.7, solid_capstyle='round', zorder=6)

    # Group separators and labels
    groups_seen = []
    for i, (tid, grp) in enumerate(TRIALS):
        if not groups_seen or groups_seen[-1][0] != grp:
            groups_seen.append((grp, i))

    for idx, (grp, start) in enumerate(groups_seen):
        if idx + 1 < len(groups_seen):
            end = groups_seen[idx + 1][1]
        else:
            end = n_trials
        mid = (start + end - 1) / 2.0

        ax.text(-0.03, mid, grp, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=TS + 1,
                fontweight='bold', color=GROUP_COLORS.get(grp, 'black'))

        if idx > 0:
            ax.axhline(start - 0.5, color='white', lw=1.5, zorder=4)

    # Row count labels
    current_grp = None
    count = 0
    for i, (tid, grp) in enumerate(TRIALS):
        if grp != current_grp:
            current_grp = grp
            count = 1
        else:
            count += 1
        ax.text(1.02, i, str(count), transform=ax.get_yaxis_transform(),
                ha='left', va='center', fontsize=TS - 1,
                color='#999999')

    # Axes
    ax.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=4)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(-0.5, n_trials - 0.5)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=TS)

    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.88, 0.25, 0.025, 0.45])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r'$R^* = R\,/\,R_{\mathrm{far}}$', fontsize=LS, labelpad=6)
    cbar.ax.tick_params(labelsize=TS)
    cbar.outline.set_linewidth(LW)

    # Delta legend
    legend_y = n_trials - 0.5
    ax.plot([2.85, 2.85], [legend_y - 0.4, legend_y + 0.4],
            color='red', lw=0.7, solid_capstyle='round', zorder=6,
            clip_on=False)
    ax.text(2.92, legend_y, r'$= \delta$', fontsize=TS,
            va='center', ha='left', color='black', clip_on=False)

    # Panel label
    ax.text(-0.16, 1.02, 'C', transform=ax.transAxes,
            fontsize=12.0, fontweight='bold', va='top')

    # Save
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'heatmap_all_trials{ext}', **kw)
    plt.close(fig)
    print(f"\nSaved to {OUTPUT_DIR}/heatmap_all_trials.*")


if __name__ == '__main__':
    main()
