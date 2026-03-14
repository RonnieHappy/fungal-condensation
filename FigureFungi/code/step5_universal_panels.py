#!/usr/bin/env python3
"""Universal mechanism panels: survival gradient vs drying power, size gradient, d2-law."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from lifelines import KaplanMeierFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paths
THIS_DIR    = Path(__file__).parent
OUTPUT_DIR  = THIS_DIR.parent / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT   = THIS_DIR.resolve().parent.parent
TRACK_DIR   = REPO_ROOT / 'FigureHGAggregate' / 'code' / 'test_tracking' / 'output'
HG_METRICS  = REPO_ROOT / 'FigureHGAggregate' / 'output' / 'hydrogel_metrics.csv'
F_METRICS   = OUTPUT_DIR / 'fungi_metrics.csv'

# All tracked trials (30: 15 HG + 15 fungi)
ALL_TRIALS = {
    # Hydrogels
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
    # Fungi
    'Green.1': 'Green', 'Green.2': 'Green', 'Green.3': 'Green',
    'Green.4': 'Green', 'Green.5': 'Green',
    'white.1': 'White', 'white.2': 'White', 'white.3': 'White',
    'white.4': 'White', 'white.5': 'White',
    'black.1': 'Black', 'black.2': 'Black', 'black.3': 'Black',
    'black.4': 'Black', 'black.5': 'Black',
}

# Colors and markers
COLORS = {
    'Agar': '#3A9E6F', '1:1': '#3A6FBF', '2:1': '#C0392B',
    'Green': '#4CAF50', 'White': '#9E9E9E', 'Black': '#212121',
}
EDGE = {
    'Agar': '#2E7D32', '1:1': '#2C5F9F', '2:1': '#922B21',
    'Green': '#2E7D32', 'White': '#616161', 'Black': '#000000',
}
MARKER = {                         # circles for HG, diamonds for fungi
    'Agar': 'o', '1:1': 'o', '2:1': 'o',
    'Green': 'D', 'White': 'D', 'Black': 'D',
}
GROUP_ORDER = ['Agar', '1:1', '2:1', 'Green', 'White', 'Black']

# Plot style
MM         = 1 / 25.4
TICK_SIZE  = 7.0
LABEL_SIZE = 8.5
PANEL_LBL  = 12.0
LW         = 0.6
LW_DATA    = 0.8

MIN_FRAMES = 3
DIST_BIN   = 200       # µm
T_SEED     = 900       # s (15 min)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': TICK_SIZE,
    'axes.linewidth': LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': LW_DATA,
    'svg.fonttype': 'none',
})


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE, pad=2)


def _save(fig, stem):
    for ext in ('.svg', '.pdf', '.png'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'{stem}{ext}', **kw)
    plt.close(fig)
    print(f'  Saved -> {OUTPUT_DIR}/{stem}.*')


# --- Data helpers ---

def load_trial(trial_id):
    """Load tracked histories, compute forward lifetime from seed."""
    path = TRACK_DIR / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()
    return df


def tau50_profile(trial_id, size_range=None, min_per_bin=15):
    """KM tau50 per 200 um distance bin (forward lifetime)."""
    df = load_trial(trial_id)
    if df is None:
        return None, None
    if size_range is not None:
        df = df.dropna(subset=['R_eq_seed']).copy()
        df = df[(df['R_eq_seed'] >= size_range[0]) &
                (df['R_eq_seed'] <= size_range[1])].copy()
        if len(df) < 30:
            return None, None
    d_max = df['distance_um'].max()
    bins = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < min_per_bin:
            continue
        center = bins[int(b)] + DIST_BIN / 2
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center / 1000.0)      # mm
            tau_vals.append(t50)
    if not d_vals:
        return None, None
    return np.array(d_vals), np.array(tau_vals)


def get_iqr_band(trial_id):
    """Return (Q25, Q75) of R_eq_seed for size matching."""
    df = load_trial(trial_id)
    if df is None:
        return None
    r = df['R_eq_seed'].dropna()
    r = r[r > 0]
    if len(r) < 30:
        return None
    return (r.quantile(0.25), r.quantile(0.75))


def get_delta_map():
    """Build trial_id -> delta (um) from both metrics CSVs."""
    delta = {}
    hg = pd.read_csv(HG_METRICS)
    delta.update(dict(zip(hg['trial_id'], hg['delta_um'])))
    fm = pd.read_csv(F_METRICS)
    delta.update(dict(zip(fm['trial_id'], fm['delta_um'])))
    # Green.2: not in fungi_metrics (never processed); estimate from P5 of
    # distance distribution matching the raycast method used for other trials.
    return delta


def _dedup_legend(ax, loc='upper left', ncol=2, extra_handles=None):
    """De-duplicate legend entries and order by GROUP_ORDER."""
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_h, ordered_l = [], []
    for l in GROUP_ORDER:
        if l in by_label:
            ordered_h.append(by_label[l])
            ordered_l.append(l)
    if extra_handles:
        for h, l in extra_handles:
            ordered_h.append(h)
            ordered_l.append(l)
    if not ordered_l:
        return
    ax.legend(ordered_h, ordered_l,
              fontsize=TICK_SIZE - 0.5, loc=loc, frameon=False,
              labelspacing=0.3, handlelength=1.2, handletextpad=0.4,
              ncol=ncol, columnspacing=0.8)


# --- Panel G: dtau50/dr vs delta ---

def make_panel_G():
    delta_map = get_delta_map()

    fig, ax = plt.subplots(figsize=(75 * MM, 68 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.20)

    # Collect per-trial values
    groups     = {g: {'x': [], 'y': []} for g in GROUP_ORDER}
    groups_sm  = {g: {'x': [], 'y': []} for g in GROUP_ORDER}

    for tid, grp in ALL_TRIALS.items():
        if tid not in delta_map:
            continue
        delta = delta_map[tid]

        # Overall
        d, t = tau50_profile(tid, size_range=None, min_per_bin=10)
        if d is not None and len(d) >= 5:
            sl = stats.linregress(d, t).slope
            groups[grp]['x'].append(delta)
            groups[grp]['y'].append(sl)
            print(f'  G  {tid:12s}: delta={delta:.0f} um, '
                  f'dtau/dr={sl:.2f} min/mm')

        # Size-matched (IQR)
        band = get_iqr_band(tid)
        if band is not None:
            d2, t2 = tau50_profile(tid, size_range=band, min_per_bin=10)
            if d2 is not None and len(d2) >= 5:
                sl2 = stats.linregress(d2, t2).slope
                groups_sm[grp]['x'].append(delta)
                groups_sm[grp]['y'].append(sl2)

    # Individual trials
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if not gd['x']:
            continue
        ax.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=18,
                   color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

    # Group means +/- SEM
    HG_GROUPS  = {'Agar', '1:1', '2:1'}
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if len(gd['x']) < 2:
            continue
        xm, ym = np.mean(gd['x']), np.mean(gd['y'])
        xe, ye = stats.sem(gd['x']), stats.sem(gd['y'])
        fmt = 'o' if grp in HG_GROUPS else 'D'
        ms  = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.errorbar(xm, ym, xerr=xe, yerr=ye,
                    fmt=fmt, color=COLORS[grp], markersize=ms,
                    markeredgecolor=mec, markeredgewidth=mew,
                    capsize=2.5, capthick=LW, elinewidth=LW,
                    ecolor=EDGE[grp], label=grp, zorder=5)

    _dedup_legend(ax, loc='upper left')

    # Overall regression
    xa = np.concatenate([groups[g]['x'] for g in GROUP_ORDER])
    ya = np.concatenate([groups[g]['y'] for g in GROUP_ORDER])
    valid = ~np.isnan(ya)
    xfit = np.linspace(0, 1100, 100)
    if valid.sum() >= 3:
        res = stats.linregress(xa[valid], ya[valid])
        r2 = res.rvalue ** 2
        ax.plot(xfit, res.intercept + res.slope * xfit,
                '-', color='#333333', lw=LW_DATA, alpha=0.7, zorder=2)
        print(f'\n  Panel G overall: R2={r2:.3f}, slope={res.slope:.5f}, '
              f'n={valid.sum()}')

    # Size-matched regression
    xa_sm = np.concatenate([groups_sm[g]['x'] for g in GROUP_ORDER])
    ya_sm = np.concatenate([groups_sm[g]['y'] for g in GROUP_ORDER])
    valid_sm = ~np.isnan(ya_sm)
    if valid_sm.sum() >= 3:
        res_sm = stats.linregress(xa_sm[valid_sm], ya_sm[valid_sm])
        r2_sm = res_sm.rvalue ** 2
        ax.plot(xfit, res_sm.intercept + res_sm.slope * xfit,
                '-', color='#C0392B', lw=LW_DATA, alpha=0.7, zorder=2)
        print(f'  Panel G size-matched: R2={r2_sm:.3f}, '
              f'slope={res_sm.slope:.5f}, n={valid_sm.sum()}')

    # Annotations
    ax.text(0.95, 0.18,
            f'Overall $R^2$ = {r2:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 0.5, color='#333333')
    ax.text(0.95, 0.08,
            f'Size-matched $R^2$ = {r2_sm:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 0.5, color='#C0392B')

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'd$\tau_{50}$/d$r$ (min mm$^{-1}$)',
                  fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)
    ax.text(-0.14, 1.04, 'D', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_D')


H_BIN_UM   = 100          # bin width (µm)
H_MIN_DROP = 3            # minimum droplets per bin
H_T_WINDOW = (14.5, 15.5)
# Zone definitions
N_NEAR_BINS = 5           # first 5 bins (0–500 µm); empty bins = R=0
D_MID_LO    = 900;  D_MID_HI  = 1100   # mid zone
D_FAR_LO    = 1900; D_FAR_HI  = 2100   # far zone

# EDT data paths
HG_AGG_DIR  = REPO_ROOT / 'FigureHGAggregate' / 'raw_data' / 'aggregate_edt'
F_AGG_DIR   = REPO_ROOT / 'FigureFungi' / 'raw_data' / 'aggregate_edt'


def _load_edt_droplets(trial_id):
    """Load EDT snapshot droplets in the 14.5-15.5 min window."""
    for d in (HG_AGG_DIR, F_AGG_DIR):
        p = d / f'{trial_id}_edt_droplets.csv'
        if p.exists():
            df = pd.read_csv(p)
            tw = df[(df['time_min'] >= H_T_WINDOW[0]) &
                    (df['time_min'] <= H_T_WINDOW[1]) &
                    (df['radius_um'] > 0)].copy()
            return tw
    return None


def _size_gradient(trial_id):
    """Compute size gradient from binned means.

    Near: mean of first N_NEAR_BINS bins (0–500 µm).  Empty bins count as
          R = 0 because the dry zone *is* the suppression signal.
    Mid:  binned mean in 900–1100 µm.
    Far:  binned mean in 1900–2100 µm.
    Returns (R_far − R_near) / R_mid.
    """
    tw = _load_edt_droplets(trial_id)
    if tw is None or len(tw) < 30:
        return None

    max_d = tw['distance_um'].max()
    bins = np.arange(0, max_d + H_BIN_UM, H_BIN_UM)
    tw = tw.copy()
    tw['bin'] = pd.cut(tw['distance_um'], bins=bins,
                       labels=bins[:-1] + H_BIN_UM / 2).astype(float)
    grp = tw.groupby('bin')['radius_um'].mean()

    # Near: first N_NEAR_BINS bins — empty = 0
    near_centers = [H_BIN_UM / 2 + i * H_BIN_UM for i in range(N_NEAR_BINS)]
    near_vals = [grp[c] if c in grp.index else 0.0 for c in near_centers]
    R_near = np.mean(near_vals)

    # Mid
    mid_vals = grp[(grp.index >= D_MID_LO) & (grp.index <= D_MID_HI)]
    if len(mid_vals) < 1:
        return None
    R_mid = mid_vals.mean()

    # Far
    far_vals = grp[(grp.index >= D_FAR_LO) & (grp.index <= D_FAR_HI)]
    if len(far_vals) < 1:
        return None
    R_far = far_vals.mean()

    if R_mid <= 0:
        return None

    return (R_far - R_near) / R_mid




def make_panel_H():
    delta_map = get_delta_map()

    fig, ax = plt.subplots(figsize=(75 * MM, 68 * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.93, bottom=0.20)

    groups = {g: {'x': [], 'y': []} for g in GROUP_ORDER}

    for tid, grp in ALL_TRIALS.items():
        if tid not in delta_map:
            continue
        result = _size_gradient(tid)
        if result is None:
            continue
        delta = delta_map[tid]
        groups[grp]['x'].append(delta)
        groups[grp]['y'].append(result)
        print(f'  H  {tid:12s}: delta={delta:.0f} um, '
              f'size_grad={result:+.3f}')

    # Individual trials
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if not gd['x']:
            continue
        ax.scatter(gd['x'], gd['y'], marker=MARKER[grp], s=18,
                   color=COLORS[grp], alpha=0.25, edgecolors='none', zorder=3)

    # Group means +/- SEM
    HG_GROUPS = {'Agar', '1:1', '2:1'}
    for grp in GROUP_ORDER:
        gd = groups[grp]
        if len(gd['x']) < 2:
            continue
        xm, ym = np.mean(gd['x']), np.mean(gd['y'])
        xe, ye = stats.sem(gd['x']), stats.sem(gd['y'])
        fmt = 'o' if grp in HG_GROUPS else 'D'
        ms  = 6.0 if grp in HG_GROUPS else 5.0
        mec = 'white' if grp in HG_GROUPS else EDGE[grp]
        mew = 0.4 if grp in HG_GROUPS else 0.5
        ax.errorbar(xm, ym, xerr=xe, yerr=ye,
                    fmt=fmt, color=COLORS[grp], markersize=ms,
                    markeredgecolor=mec, markeredgewidth=mew,
                    capsize=2.5, capthick=LW, elinewidth=LW,
                    ecolor=EDGE[grp], label=grp, zorder=5)

    _dedup_legend(ax, loc='upper left')

    # Regression
    xa = np.concatenate([groups[g]['x'] for g in GROUP_ORDER])
    ya = np.concatenate([groups[g]['y'] for g in GROUP_ORDER])
    valid = ~np.isnan(ya)
    if valid.sum() >= 3:
        res = stats.linregress(xa[valid], ya[valid])
        r2 = res.rvalue ** 2
        xfit = np.linspace(0, 1100, 100)
        ax.plot(xfit, res.intercept + res.slope * xfit,
                '-', color='#333333', lw=LW_DATA, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08,
                f'$R^2$ = {r2:.2f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=TICK_SIZE - 0.5)
        print(f'\n  Panel H: R2={r2:.3f}, slope={res.slope:.6f}, '
              f'n={valid.sum()}')

    ax.set_xlabel(r'$\delta$ ($\mu$m)', fontsize=LABEL_SIZE, labelpad=3)
    ax.set_ylabel(r'$(R_{\mathrm{far}} - R_{\mathrm{near}})\,/\,R_{\mathrm{mid}}$',
                  fontsize=LABEL_SIZE, labelpad=3)
    ax.set_xlim(0, 1100)
    style_ax(ax)
    ax.text(-0.14, 1.04, 'E', transform=ax.transAxes,
            fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_E')


# --- Panel I: d2-law scatter ---

NEAR_UM = 800
FAR_UM  = 1500
C_NEAR  = '#E57373'     # coral
C_FAR   = '#42A5F5'     # blue


def _plot_trial_d2(ax, trial_id, label_text):
    """Scatter tau vs R0^2 for one trial, colored by near/far distance."""
    df = load_trial(trial_id)
    if df is None:
        return
    # Uncensored droplets with valid initial size
    df = df[(~df['censored']) &
            df['R_eq_seed'].notna() &
            (df['R_eq_seed'] > 0)].copy()
    df['R2'] = df['R_eq_seed'] ** 2

    # Overall correlation
    r_all = np.corrcoef(df['R2'], df['tau_fwd_min'])[0, 1]
    print(f'  I  {trial_id} overall: n={len(df)}, r={r_all:.3f}')

    near = df[df['distance_um'] < NEAR_UM]
    far  = df[df['distance_um'] > FAR_UM]
    mid  = df[(df['distance_um'] >= NEAR_UM) &
              (df['distance_um'] <= FAR_UM)]

    grp = ALL_TRIALS[trial_id]
    mk  = MARKER[grp]

    # Middle range (background)
    if len(mid) > 0:
        ax.scatter(mid['R2'], mid['tau_fwd_min'], c='#BDBDBD', s=3,
                   alpha=0.12, edgecolors='none', marker=mk,
                   zorder=1, rasterized=True)

    for sub, color, lbl in [
        (near, C_NEAR, f'$d$ < {NEAR_UM / 1000:.1f} mm'),
        (far,  C_FAR,  f'$d$ > {FAR_UM / 1000:.1f} mm'),
    ]:
        if len(sub) < 5:
            continue
        ax.scatter(sub['R2'], sub['tau_fwd_min'], c=color, s=3,
                   alpha=0.15, edgecolors='none', marker=mk,
                   zorder=2, rasterized=True)
        # Regression line
        res = stats.linregress(sub['R2'], sub['tau_fwd_min'])
        xfit = np.linspace(sub['R2'].min(), sub['R2'].max(), 100)
        ax.plot(xfit, res.intercept + res.slope * xfit, color=color,
                lw=LW_DATA, alpha=0.9, zorder=3, label=lbl)
        print(f'  I  {trial_id} {lbl}: n={len(sub)}, '
              f'r={res.rvalue:.3f}')

    ax.set_title(label_text, fontsize=LABEL_SIZE, pad=3)
    ax.set_xlabel(r'$R_0^2$ ($\mu$m$^2$)', fontsize=LABEL_SIZE, labelpad=3)
    ax.text(0.95, 0.08,
            f'$r$ = {r_all:.2f}',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=TICK_SIZE - 0.5)
    ax.legend(fontsize=TICK_SIZE - 0.5, loc='upper left', frameon=False,
              labelspacing=0.3, handlelength=1.2, handletextpad=0.4)
    style_ax(ax)


def make_panel_I():
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(150 * MM, 68 * MM))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.90, bottom=0.18,
                        wspace=0.30)

    ax_l.set_ylabel(r'$\tau$ (min)', fontsize=LABEL_SIZE, labelpad=3)
    _plot_trial_d2(ax_l,  '2to1.1',  '2:1 NaCl')
    _plot_trial_d2(ax_r, 'Green.1', 'Green fungus')
    ax_l.text(-0.14, 1.06, 'I', transform=ax_l.transAxes,
              fontsize=PANEL_LBL, fontweight='bold', va='top')

    _save(fig, 'panel_I')


# --- Main ---

def main():
    print('=== Generating panels G-I (universal mechanism) ===\n')

    make_panel_G()
    print()
    make_panel_H()
    print()
    make_panel_I()

    print('\nDone.')


if __name__ == '__main__':
    main()
