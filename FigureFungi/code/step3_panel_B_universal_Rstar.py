#!/usr/bin/env python3
"""Universal R*(d) overlay for all 30 trials across 6 groups."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPO_ROOT  = THIS_DIR.resolve().parent.parent
FUNGI_AGG  = REPO_ROOT / 'FigureFungi' / 'raw_data' / 'aggregate_edt'
HG_AGG     = REPO_ROOT / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'

# Plot style
MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
LW = 0.6

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          TS,
    'axes.linewidth':     LW,
    'xtick.major.width':  LW, 'ytick.major.width': LW,
    'xtick.major.size':   3.5, 'ytick.major.size': 3.5,
    'xtick.direction':    'out', 'ytick.direction': 'out',
    'svg.fonttype':       'none',
})

# Colors
COLORS = {
    'Agar':  '#9E9E9E',
    '1:1':   '#5B8FC9',
    '2:1':   '#C0392B',
    'Green': '#4CAF50',
    'White': '#757575',
    'Black': '#212121',
}

# All 30 trials
ALL_TRIALS = {
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.2': 'White', 'white.3': 'White',
    'white.4': 'White', 'white.5': 'White',
    'black.1': 'Black', 'black.2': 'Black', 'black.3': 'Black',
    'black.4': 'Black', 'black.5': 'Black',
}

# Parameters
T_WINDOW     = (14.5, 15.5)
BIN_WIDTH_UM = 200
MIN_DROPS    = 5
DIST_MAX_MM  = 2.5
FAR_THRESH   = 1500  # µm — bins beyond this define R_far


def load_profile(trial_id):
    """Load binned stats, extract R*(d) profile at T_WINDOW."""
    for agg_dir in [FUNGI_AGG, HG_AGG]:
        path = agg_dir / f'{trial_id}_edt_binned_statistics.csv'
        if path.exists():
            break
    else:
        return None

    df = pd.read_csv(path)
    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1])].copy()

    profile = []
    for d_bin, grp in tw.groupby('distance_bin_um'):
        n = grp['n_droplets'].sum()
        if n >= MIN_DROPS:
            wmean = np.average(grp['mean_radius_um'].values,
                               weights=grp['n_droplets'].values)
            profile.append({'distance_mm': d_bin / 1000.0, 'R_um': wmean})

    if len(profile) < 3:
        return None

    prof_df = pd.DataFrame(profile)
    far = prof_df[prof_df['distance_mm'] > FAR_THRESH / 1000.0]
    if len(far) < 1:
        return None
    R_far = far['R_um'].mean()
    if R_far <= 0:
        return None
    prof_df['R_star'] = prof_df['R_um'] / R_far
    return prof_df


def main():
    group_order = ['2:1', '1:1', 'Agar', 'Green', 'White', 'Black']
    group_profiles = {g: [] for g in group_order}

    for tid, grp in ALL_TRIALS.items():
        prof = load_profile(tid)
        if prof is not None:
            group_profiles[grp].append(prof)
            print(f'  {tid}: {len(prof)} bins')
        else:
            print(f'  [SKIP] {tid}')

    # Figure
    fig, ax = plt.subplots(1, 1, figsize=(90 * MM, 70 * MM))
    fig.subplots_adjust(left=0.16, right=0.96, top=0.92, bottom=0.16)

    LABEL_MAP = {
        'Agar': 'Agar (control)',
        '1:1': '1:1 NaCl',
        '2:1': '2:1 NaCl',
        'Green': 'Green fungus',
        'White': 'White fungus',
        'Black': 'Black fungus',
    }

    for grp in group_order:
        frames = group_profiles[grp]
        if not frames:
            continue
        color = COLORS[grp]

        # Individual trial traces (thin, transparent)
        for prof in frames:
            mask = prof['distance_mm'] <= DIST_MAX_MM
            ax.plot(prof.loc[mask, 'distance_mm'],
                    prof.loc[mask, 'R_star'],
                    color=color, lw=0.5, alpha=0.25, zorder=1)

        # Group mean ± SEM
        combined = pd.concat(frames)
        agg = (combined[combined['distance_mm'] <= DIST_MAX_MM]
               .groupby('distance_mm')['R_star']
               .agg(['mean', 'sem']).reset_index())

        ax.plot(agg['distance_mm'], agg['mean'],
                color=color, lw=1.8, label=LABEL_MAP[grp], zorder=3)
        ax.fill_between(agg['distance_mm'],
                        agg['mean'] - agg['sem'],
                        agg['mean'] + agg['sem'],
                        color=color, alpha=0.15, zorder=2)

    ax.axhline(1.0, color='gray', ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Distance from source (mm)', fontsize=LS)
    ax.set_ylabel(r'$R^* = R\,/\,R_{\mathrm{far}}$', fontsize=LS)
    ax.set_xlim(0, DIST_MAX_MM)
    ax.legend(fontsize=TS - 1, framealpha=0.9, loc='lower right',
              handletextpad=0.4, borderpad=0.3, labelspacing=0.3)

    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    ax.text(-0.16, 1.05, 'B', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')

    # Save
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'panel_B_universal_Rstar{ext}', **kw)
    plt.close(fig)
    print(f'\nSaved to {OUTPUT_DIR}/panel_B_universal_Rstar.*')


if __name__ == '__main__':
    main()
