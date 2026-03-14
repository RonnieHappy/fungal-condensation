#!/usr/bin/env python3
"""Compute condensation metrics (delta, max_slope) for each fungal trial via tanh fit."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

# Paths
THIS_DIR   = Path(__file__).parent
AGG_DIR    = THIS_DIR.parent / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trial configuration
TRIALS = {
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.2': 'White', 'white.3': 'White',
    'white.4': 'White', 'white.5': 'White',
    'black.1': 'Black', 'black.2': 'Black', 'black.3': 'Black',
    'black.4': 'Black', 'black.5': 'Black',
}

# Raycast delta values (um)
# Pre-computed from per-droplet CSVs via step1_batch_process.py.
# Method: sample 100 points along source boundary, find nearest droplet
# to each, take mean of per-ray minimum distances.
DELTA_RAYCAST = {
    'Green.1':   279.5,  'Green.2':   316.0,  'Green.3': 297.8,
    'Green.4':   285.6,  'Green.5':   311.7,
    'white.1':   198.5,  'white.2':   126.2,  'white.3':     120.0,
    'white.4':   131.6,  'white.5':   123.0,
    'black.1':   125.4,  'black.2':    98.0,  'black.3':     111.7,
    'black.4': 135.7,  'black.5':  78.3,
}

# Analysis parameters
T_WINDOW      = (14.5, 15.5)
BIN_WIDTH_UM  = 100
MIN_DROPS_BIN = 5


# --- Core computation ---

def compute_trial_metrics(trial_id):
    """Compute delta and max_slope for one trial via tanh fit to per-droplet data."""
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Analysis window
    tw = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1])].copy()
    if len(tw) < 50:
        return None

    # Delta from raycast lookup
    delta = DELTA_RAYCAST.get(trial_id, np.nan)

    # Bin per-droplet data
    bins = np.arange(0, tw['distance_um'].max() + BIN_WIDTH_UM, BIN_WIDTH_UM)
    tw['distance_bin_um'] = pd.cut(
        tw['distance_um'], bins=bins,
        labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float)
    grouped = tw.groupby('distance_bin_um')['radius_um']
    prof = grouped.agg(mean_radius_um='mean', n_droplets='count').reset_index()
    prof = prof[prof['n_droplets'] >= MIN_DROPS_BIN].sort_values('distance_bin_um')

    if len(prof) < 5:
        return None

    x = prof['distance_bin_um'].values
    y = prof['mean_radius_um'].values

    # Tanh fit
    def tanh_model(d, y_near, y_far, alpha, r0):
        return (y_near + y_far) / 2 + (y_far - y_near) / 2 * np.tanh(alpha * (d - r0))

    y_floor = max(1.0, y.min() * 0.3)
    p0 = [y.min(), y.max(), 0.002, np.median(x)]
    bounds = ([y_floor, y_floor, 1e-6, x.min()],
              [np.inf, np.inf, 1.0, x.max()])

    try:
        popt, pcov = curve_fit(tanh_model, x, y, p0=p0, bounds=bounds, maxfev=10000)
        y_near, y_far, alpha, r0 = popt
    except RuntimeError:
        return {'delta_um': delta}

    max_slope = alpha * (y_far - y_near) / 2.0
    transition_width = 2.0 / alpha if alpha > 0 else np.nan

    # r0 undefined when transition spans observation window
    obs_range = x.max() - x.min()
    r0_out = np.nan if transition_width > 0.75 * obs_range else r0

    return {
        'delta_um':            delta,
        'max_slope':           max_slope,
        'r0_um':               r0_out,
        'alpha':               alpha,
        'y_near':              y_near,
        'y_far':               y_far,
        'transition_width_um': transition_width,
    }


def main():
    print("Computing fungal metrics (tanh fit on absolute distance)")
    print(f'  t = {T_WINDOW[0]}–{T_WINDOW[1]} min, bin = {BIN_WIDTH_UM} µm\n')

    rows = []

    for tid, species in TRIALS.items():
        result = compute_trial_metrics(tid)
        if result is None:
            print(f'  [SKIP] {tid}: not enough data')
            continue

        row = {'trial_id': tid, 'species': species, **result}
        rows.append(row)

        ms_str = f'{result["max_slope"]:.5f}' if not np.isnan(result.get('max_slope', np.nan)) else '  N/A'

        print(f'  {tid:<14} ({species:<6})  '
              f'delta={result["delta_um"]:6.1f}  '
              f'max_slope={ms_str}')

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / 'fungi_metrics.csv'
    df.to_csv(out, index=False)
    print(f'\n-> saved {out}')

    # Species summary
    print('\n── Species means ──')
    for sp in ['Green', 'White', 'Black']:
        g = df[df['species'] == sp]
        n = len(g)
        d  = g['delta_um']
        ms = g['max_slope'].dropna()
        print(f'  {sp:<6}: n={n}  '
              f'delta={d.mean():.0f}+/-{d.sem():.0f}  '
              f'max_slope={ms.mean():.5f}+/-{ms.sem():.5f}')


if __name__ == '__main__':
    main()
