#!/usr/bin/env python3
"""Compute Beysens growth profiles and generate per-trial figures."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
THIS_DIR   = Path(__file__).parent
AGG_DIR    = THIS_DIR.parent / 'raw_data' / 'aggregate_edt'
OUTPUT_DIR = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
BIN_WIDTH_UM     = 300
MIN_DROPLETS_BIN = 5
MIN_FRAMES_FIT   = 5
T_DATA_MIN       = 5.0    # start of analysis window (min)
T_DATA_MAX       = 15.0   # end of analysis window (min)

# Trial registry
# trial_id: delta_um (dry-zone radius in µm; 0 for fungi since EDT is from polygon)
TRIALS = {
    '2to1.4': 900,
}


# Geometry helpers

def polygon_area_perimeter(xs, ys):
    n = len(xs)
    area = 0.5 * abs(sum(xs[i]*ys[(i+1)%n] - xs[(i+1)%n]*ys[i] for i in range(n)))
    perim = sum(np.hypot(xs[(i+1)%n]-xs[i], ys[(i+1)%n]-ys[i]) for i in range(n))
    return area, perim


def steiner_bin_area(P_body, delta_um, bin_lo, bin_hi):
    """Area of annular strip via Steiner parallel-body formula."""
    r_lo = delta_um + bin_lo
    r_hi = delta_um + bin_hi
    return P_body * (r_hi - r_lo) + np.pi * (r_hi**2 - r_lo**2)


# Analysis

def load_droplets(tid):
    path = AGG_DIR / f'{tid}_edt_droplets.csv'
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_boundary(tid):
    path = AGG_DIR / f'{tid}_boundary_polygon.csv'
    if not path.exists():
        return None, None
    poly = pd.read_csv(path)
    _, perim = polygon_area_perimeter(poly['x'].values, poly['y'].values)
    return perim, None


def compute_profile(droplets, delta_um, P_body):
    """Compute β(r′) and ε(r′) profiles via OLS log-log fit."""
    droplets = droplets.copy()
    droplets['r_prime_um'] = droplets['distance_um'] - delta_um

    half = BIN_WIDTH_UM / 2
    bin_centers = np.arange(0, 3600, BIN_WIDTH_UM)
    bin_edges   = bin_centers - half

    ts_rows, profile_rows = [], []

    for j, bc in enumerate(bin_centers):
        lo, hi = bin_edges[j], bin_edges[j] + BIN_WIDTH_UM
        in_bin = droplets[(droplets['r_prime_um'] >= lo) & (droplets['r_prime_um'] < hi)]
        if in_bin.empty:
            continue

        A_bin = steiner_bin_area(P_body, delta_um, lo, hi) if P_body else np.nan

        bin_ts = []
        for t, frame in in_bin.groupby('time_min'):
            R = frame['radius_um'].values
            R = R[R > 0]
            if len(R) < MIN_DROPLETS_BIN:
                continue
            R_med = np.median(R)
            eps = np.sum(np.pi * R**2) / A_bin if not np.isnan(A_bin) and A_bin > 0 else np.nan
            ts_rows.append({'r_prime_um': bc, 'time_min': t,
                            'R_median': R_med, 'n_droplets': len(R), 'epsilon': eps})
            bin_ts.append((t, R_med))

        if len(bin_ts) < MIN_FRAMES_FIT:
            continue

        # Filter to analysis window
        bin_ts_win = [(t, R) for t, R in bin_ts if T_DATA_MIN <= t <= T_DATA_MAX]
        if len(bin_ts_win) < MIN_FRAMES_FIT:
            continue

        t_arr = np.array([x[0] for x in bin_ts_win])
        R_arr = np.array([x[1] for x in bin_ts_win])

        # Trim onset if V-shape detected
        idx_min = np.argmin(R_arr)
        if idx_min > 0:
            t_arr, R_arr = t_arr[idx_min:], R_arr[idx_min:]
        if len(t_arr) < MIN_FRAMES_FIT:
            continue

        # MAD outlier filter
        med_R = np.median(R_arr)
        mad_R = np.median(np.abs(R_arr - med_R))
        keep = np.abs(R_arr - med_R) <= 3 * mad_R
        t_arr, R_arr = t_arr[keep], R_arr[keep]
        if len(t_arr) < MIN_FRAMES_FIT:
            continue

        # OLS log-log fit: log R = β·log t + log A
        lt, lR = np.log(t_arr), np.log(R_arr)
        beta, logA, r_val, p_val, se = stats.linregress(lt, lR)
        A_fit = np.exp(logA)

        # ε at end of window
        eps_final = np.nanmedian([row['epsilon'] for row in ts_rows
                                  if row['r_prime_um'] == bc and
                                  T_DATA_MIN <= row['time_min'] <= T_DATA_MAX])

        profile_rows.append({'r_prime_um': bc, 'beta': beta, 'A': A_fit,
                              'beta_se': se, 'r2': r_val**2, 'n_frames': len(t_arr),
                              'epsilon_final': eps_final})

    return pd.DataFrame(profile_rows), pd.DataFrame(ts_rows)


# Plotting

def plot_trial(profile, ts, tid, delta_um, out_path):
    T_MIN, T_MAX = T_DATA_MIN, T_DATA_MAX
    R_MAX = 70;  R_anchor = 20.0;  t_anchor = 5.0
    t_ref = np.linspace(T_MIN, T_MAX, 200)

    r_bins   = sorted(profile['r_prime_um'].unique())
    if len(r_bins) < 3:
        print(f'  {tid}: not enough bins for plot ({len(r_bins)})')
        return
    near_bin = r_bins[0]
    far_bin  = r_bins[-1]
    mid_bin  = r_bins[len(r_bins) // 2]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5),
                              gridspec_kw={'width_ratios': [1, 1, 1, 0.9, 1.5]})
    fig.suptitle(f"{tid}  |  δ={delta_um} μm", fontsize=11, y=1.01)

    # Panels 0–2: R(t) for near/mid/far
    for ax_idx, (bc, lbl) in enumerate([(near_bin, 'Near  r′=0'),
                                         (mid_bin,  'Mid'),
                                         (far_bin,  'Far')]):
        ax = axes[ax_idx]
        sub = ts[(ts['r_prime_um'] == bc) &
                 (ts['time_min'] >= T_MIN) & (ts['time_min'] <= T_MAX)
                 ].sort_values('time_min')
        ax.plot(sub['time_min'], sub['R_median'], 'o-', color='steelblue',
                ms=4, lw=1.3, zorder=3)
        row = profile[profile['r_prime_um'] == bc]
        if not row.empty:
            beta_v = float(row['beta'].values[0])
            A_v    = float(row['A'].values[0])
            ax.plot(t_ref, A_v * t_ref**beta_v, color='steelblue', ls='-', lw=1.6, alpha=0.45)
            ax.text(0.97, 0.07, f'β = {beta_v:.2f}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=10, fontweight='bold', color='steelblue')
        ax.plot(t_ref, R_anchor*(t_ref/t_anchor)**(1/3), 'k--', lw=1.2, alpha=0.7, label='β=⅓')
        ax.plot(t_ref, R_anchor*(t_ref/t_anchor)**1.0,   'r--', lw=1.2, alpha=0.7, label='β=1')
        ax.set_xlim(T_MIN, T_MAX);  ax.set_ylim(0, R_MAX)
        ax.set_xlabel('Time (min)', fontsize=9)
        ax.set_ylabel('⟨R⟩ (μm)', fontsize=9) if ax_idx == 0 else None
        ax.set_title(f'{lbl}\nr′={bc/1000:.1f} mm', fontsize=9)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc='upper left')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)

    # Panel 3: instantaneous β(t) overlaid for near/mid/far
    ax = axes[3]
    nf_colors = plt.cm.plasma([0.05, 0.45, 0.85])
    for (bc, lbl), col in zip([(near_bin, 'Near'), (mid_bin, 'Mid'), (far_bin, 'Far')],
                               nf_colors):
        sub = ts[(ts['r_prime_um'] == bc) &
                 (ts['time_min'] >= T_MIN) & (ts['time_min'] <= T_MAX)
                 ].sort_values('time_min')
        sub = sub[sub['R_median'] > 0]
        if len(sub) < 4:
            continue
        t_s = sub['time_min'].values
        R_s = sub['R_median'].values
        b_raw = np.gradient(np.log(R_s), np.log(t_s))
        b_sm  = pd.Series(b_raw).rolling(3, center=True, min_periods=1).mean().values
        ax.plot(t_s, b_sm, '-', color=col, lw=1.8, label=f'{lbl} r′={bc/1000:.1f}mm')
        ax.fill_between(t_s, 1/3, b_sm, where=(b_sm > 1/3), alpha=0.10, color=col)
    ax.axhline(1/3, color='#555555', ls='--', lw=1.3, label='β=⅓')
    ax.axhline(1.0, color='tomato',  ls='--', lw=1.3, label='β=1')
    ax.axhline(0,   color='#bbbbbb', ls='-',  lw=0.7)
    ax.set_xlim(T_MIN, T_MAX);  ax.set_ylim(-0.5, 2.2)
    ax.set_xlabel('Time (min)', fontsize=9)
    ax.set_ylabel('Instantaneous β', fontsize=9)
    ax.set_title('Instantaneous β(t)', fontsize=9)
    ax.legend(fontsize=6.5, loc='upper right')
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    # Panel 4: all bins R(t) colored by distance (plasma)
    ax = axes[4]
    r_sorted = sorted(ts['r_prime_um'].unique())
    cmap = plt.cm.plasma
    for i, bc in enumerate(r_sorted):
        sub = ts[(ts['r_prime_um'] == bc) &
                 (ts['time_min'] >= T_MIN) & (ts['time_min'] <= T_MAX)
                 ].sort_values('time_min')
        if len(sub) < 3:
            continue
        color = cmap(i / max(len(r_sorted)-1, 1))
        ax.plot(sub['time_min'], sub['R_median'], '-', color=color, lw=1.2, alpha=0.75)
    ax.plot(t_ref, 20*(t_ref/5)**(1/3), 'k--', lw=1.4, alpha=0.6, label='β=⅓')
    ax.plot(t_ref, 20*(t_ref/5)**1.0,   'r--', lw=1.4, alpha=0.6, label='β=1')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, r_sorted[-1]/1000))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="r′ (mm)")
    ax.set_xlabel('Time (min)', fontsize=9)
    ax.set_ylabel('⟨R⟩ (μm)', fontsize=9)
    ax.set_title('All bins — colored by r′', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(T_MIN, T_MAX);  ax.set_ylim(0, R_MAX)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    fig.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ saved {out_path.name}')


# Main

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', help='Process only this trial ID')
    args = parser.parse_args()

    trials = TRIALS
    if args.trial:
        if args.trial not in TRIALS:
            print(f'Trial "{args.trial}" not in TRIALS dict')
            return
        trials = {args.trial: TRIALS[args.trial]}

    all_profiles, all_ts = [], []

    for tid, delta_um in trials.items():
        print(f'\n=== {tid} (δ={delta_um} µm) ===')
        droplets = load_droplets(tid)
        if droplets.empty:
            print(f'  [SKIP] no droplet CSV found for {tid} in {AGG_DIR}')
            continue
        P_body, _ = load_boundary(tid)
        if P_body is None:
            # Fallback: estimate perimeter from a circle of radius delta_um
            P_body = 2 * np.pi * delta_um
            print(f'  [WARN] no boundary polygon — using circular approx P={P_body:.0f} µm')

        profile, ts = compute_profile(droplets, delta_um, P_body)
        if profile.empty:
            print(f'  [WARN] no bins with enough data')
            continue

        print(f'  {len(profile)} usable bins | '
              f'β range [{profile["beta"].min():.2f}, {profile["beta"].max():.2f}]')

        profile['trial_id'] = tid
        ts['trial_id'] = tid
        all_profiles.append(profile)
        all_ts.append(ts)

        plot_trial(profile, ts, tid, delta_um, OUTPUT_DIR / f'beysens_{tid}.png')

    if all_profiles:
        pd.concat(all_profiles).to_csv(OUTPUT_DIR / 'beysens_profile_results.csv', index=False)
        print(f'\n✓ profile results → {OUTPUT_DIR}/beysens_profile_results.csv')
    print('\nDone.')


if __name__ == '__main__':
    main()
