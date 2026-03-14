#!/usr/bin/env python3
"""Generate manuscript panels I, J, K, L from tracked droplet data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from lifelines import KaplanMeierFitter
from scipy import stats

# Paths
THIS_DIR    = Path(__file__).parent
TRACK_OUT   = THIS_DIR / 'output'                          # tracked CSVs
FIG_OUT     = THIS_DIR.parent.parent / 'output'             # figure output
HG_METRICS  = FIG_OUT / 'hydrogel_metrics.csv'

# Trial mapping
HG_TRIALS = {
    'agar.1': 'Agar', 'agar.2': 'Agar', 'agar.3': 'Agar',
    'agar.4': 'Agar', 'agar.5': 'Agar',
    '1to1.1': '1:1', '1to1.2': '1:1', '1to1.3': '1:1',
    '1to1.4': '1:1', '1to1.5': '1:1',
    '2to1.1': '2:1', '2to1.2': '2:1', '2to1.3': '2:1',
    '2to1.4': '2:1', '2to1.5': '2:1',
}
GROUP_ORDER = ['Agar', '1:1', '2:1']

# Plot style
MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
LW = 0.6
COLORS = {'Agar': '#3A9E6F', '1:1': '#5B8FC9', '2:1': '#C0392B'}
SURV_DIST_UM  = [900, 1500, 2100, 2900]
SURV_COLORS   = ['#1a237e', '#7b1fa2', '#e65100', '#f0a500']

MIN_FRAMES = 3
DIST_BIN = 200  # µm

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


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


def load_trial(trial_id):
    path = TRACK_OUT / f'{trial_id}_track_histories.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df['n_frames'] >= MIN_FRAMES].copy()
    df['lifetime_min'] = df['lifetime_s'] / 60.0
    return df


# --- Helpers ---

def tracked_tau50_profile(trial_id):
    """Return (d_mm[], tau_min[]) — KM τ₅₀ per 200 µm bin."""
    df = load_trial(trial_id)
    if df is None:
        return None, None
    d_max = df['distance_um'].max()
    bins = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df = df.copy()
    df['db'] = pd.cut(df['distance_um'], bins=bins, labels=False)
    kmf = KaplanMeierFitter()
    d_vals, tau_vals = [], []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < 15:
            continue
        center = bins[int(b)] + DIST_BIN / 2
        kmf.fit(sub['lifetime_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            d_vals.append(center / 1000.0)  # mm
            tau_vals.append(t50)
    return np.array(d_vals), np.array(tau_vals)


T_SEED = 900  # seed frame ≈ 15 min (start of evaporation phase)


def decompose_log_gradient(trial_id):
    """d²-law log decomposition of the τ₅₀ gradient.

    Under d²-law: τ = R₀²/K  →  ln τ = ln R² − ln K
    So  d(ln τ)/dr  =  d(ln R²)/dr  −  d(ln K)/dr
                    =  [size channel] + [rate channel]

    Uses forward lifetime only (from seed at t=15 min to death).
    Returns (d_lntau_dr, d_lnR2_dr, neg_d_lnK_dr) or (None, None, None).
    """
    df = load_trial(trial_id)
    if df is None:
        return None, None, None
    df = df.dropna(subset=['R_eq_seed', 'distance_um', 't_death_s']).copy()
    df = df[df['R_eq_seed'] > 0].copy()
    df['tau_fwd_min'] = (df['t_death_s'] - T_SEED) / 60.0
    df = df[df['tau_fwd_min'] > 0].copy()
    df['R2'] = df['R_eq_seed'] ** 2

    d_max = df['distance_um'].max()
    bins_e = np.arange(0, d_max + DIST_BIN, DIST_BIN)
    df['db'] = pd.cut(df['distance_um'], bins=bins_e, labels=False)

    kmf = KaplanMeierFitter()
    rows = []
    for b in sorted(df['db'].dropna().unique()):
        sub = df[df['db'] == b]
        if len(sub) < 15:
            continue
        center = bins_e[int(b)] + DIST_BIN / 2
        kmf.fit(sub['tau_fwd_min'], event_observed=~sub['censored'])
        t50 = kmf.median_survival_time_
        mean_R2 = sub['R2'].mean()
        if np.isfinite(t50) and t50 > 0 and mean_R2 > 0:
            rows.append({
                'd_mm': center / 1000.0,
                'ln_tau': np.log(t50),
                'ln_R2': np.log(mean_R2),
                'ln_K': np.log(mean_R2 / t50),
            })
    if len(rows) < 5:
        return None, None, None
    bdf = pd.DataFrame(rows)
    sl_tau = stats.linregress(bdf['d_mm'], bdf['ln_tau']).slope
    sl_R2 = stats.linregress(bdf['d_mm'], bdf['ln_R2']).slope
    sl_K = stats.linregress(bdf['d_mm'], bdf['ln_K']).slope
    return sl_tau, sl_R2, -sl_K  # total, size, rate


def compute_slope(d_mm, tau_min):
    """Linear regression slope dτ/dr (min/mm)."""
    if d_mm is None or len(d_mm) < 5:
        return None, None, None
    sl, ic, r, p, se = stats.linregress(d_mm, tau_min)
    return sl, r, p


# --- Panel I: Tracked KM survival ---

def make_panel_I():
    trials_21 = [t for t, g in HG_TRIALS.items() if g == '2:1']
    dfs = [load_trial(t) for t in trials_21]
    dfs = [d for d in dfs if d is not None]
    pool = pd.concat(dfs, ignore_index=True)

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.92, bottom=0.16)

    band_hw = 300  # ±300 µm
    kmf = KaplanMeierFitter()
    tau50_pts = []

    for tgt_um, color in zip(SURV_DIST_UM, SURV_COLORS):
        sub = pool[(pool['distance_um'] >= tgt_um - band_hw) &
                    (pool['distance_um'] <  tgt_um + band_hw)]
        if len(sub) < 20:
            continue
        label = f'{tgt_um / 1000:.1f} mm'
        kmf.fit(sub['lifetime_min'], event_observed=~sub['censored'],
                label=label)
        kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.18,
                                    linewidth=1.8)
        t50 = kmf.median_survival_time_
        if np.isfinite(t50):
            tau50_pts.append((t50, color))

    # τ₅₀ reference line
    ax.axhline(0.5, color='gray', ls='--', lw=0.6, alpha=0.6, zorder=1)
    if tau50_pts:
        ax.text(tau50_pts[0][0] - 0.3, 0.52, r'$\tau_{50}$',
                fontsize=TS, color='gray', ha='right')
    for t50, color in tau50_pts:
        ax.plot(t50, 0.5, 'o', color=color, ms=4.5,
                markeredgecolor='white', markeredgewidth=0.5, zorder=5)

    ax.set_xlabel(r'Time since birth (min)', fontsize=LS, labelpad=3)
    ax.set_ylabel('Fraction surviving', fontsize=LS, labelpad=3)
    ax.legend(fontsize=TS - 0.5, loc='upper right', framealpha=0.9,
              title='Distance from source', title_fontsize=TS - 1,
              handletextpad=0.3, borderpad=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    style_ax(ax)

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panel_survival{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'Panel I → {FIG_OUT}/panel_survival.*')


# --- Panel J: tau50 vs distance ---

def plot_panel_J(ax):
    profiles = {}
    for grp in GROUP_ORDER:
        profiles[grp] = []
    for tid, grp in HG_TRIALS.items():
        d, t = tracked_tau50_profile(tid)
        if d is not None and len(d) >= 3:
            profiles[grp].append((d, t))

    for grp in GROUP_ORDER:
        profs = profiles[grp]
        if not profs:
            continue
        all_d = sorted(set(d for dd, _ in profs for d in dd))
        means, sems, ds = [], [], []
        for d in all_d:
            vals = []
            for dd, tt in profs:
                idx = np.where(np.isclose(dd, d, atol=0.01))[0]
                if len(idx) == 1:
                    vals.append(tt[idx[0]])
            if len(vals) >= 2:
                ds.append(d)
                means.append(np.mean(vals))
                sems.append(stats.sem(vals))
        if len(ds) >= 2:
            ax.errorbar(ds, means, yerr=sems, fmt='o-',
                        color=COLORS[grp], ms=3.5, lw=1.2,
                        markeredgecolor='white', markeredgewidth=0.4,
                        capsize=2.5, capthick=LW, elinewidth=LW,
                        label=grp, zorder=3)

    # Overlay 4 survival-panel distances on 2:1 curve
    twoto1_profs = profiles['2:1']
    if twoto1_profs:
        all_d_21 = sorted(set(d for dd, _ in twoto1_profs for d in dd))
        from matplotlib.lines import Line2D
        surv_handles = []
        for tgt_um, sc in zip(SURV_DIST_UM, SURV_COLORS):
            tgt_mm = tgt_um / 1000.0
            closest = min(all_d_21, key=lambda d: abs(d - tgt_mm))
            vals = []
            for dd, tt in twoto1_profs:
                idx = np.where(np.isclose(dd, closest, atol=0.01))[0]
                if len(idx) == 1:
                    vals.append(tt[idx[0]])
            d_label = f'{closest:.1f} mm'
            if len(vals) >= 2:
                ax.errorbar(closest, np.mean(vals), yerr=stats.sem(vals),
                            fmt='o', color=sc, ms=5.5,
                            markeredgecolor='white', markeredgewidth=0.6,
                            capsize=2.5, capthick=LW, elinewidth=LW, zorder=5)
                surv_handles.append(d_label)
            elif vals:
                ax.plot(closest, vals[0], 'o', color=sc, ms=5.5,
                        markeredgecolor='white', markeredgewidth=0.6, zorder=5)
                surv_handles.append(d_label)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        combined_h = list(by_label.values())
        combined_l = list(by_label.keys())
        for dl, sc in zip(surv_handles, SURV_COLORS):
            combined_h.append(Line2D([0], [0], marker='o', color=sc, ls='none',
                                     ms=4.5, markeredgecolor='white',
                                     markeredgewidth=0.4))
            combined_l.append(dl)
        ax.legend(combined_h, combined_l,
                  fontsize=TS - 0.5, loc='lower right', framealpha=0.9,
                  handletextpad=0.3, borderpad=0.3, ncol=2,
                  title='Tracked', title_fontsize=TS - 1)
    else:
        ax.legend(fontsize=TS - 0.5, loc='lower right', framealpha=0.9)

    ax.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'Half-life $\tau_{50}$ (min)', fontsize=LS, labelpad=3)
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'I', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


# --- Panel K: dtau/dr vs (1 - a_w) ---

def plot_panel_K(ax):
    """Direct tracked replacement for the original pop-count dτ/dr panel."""
    hg = pd.read_csv(HG_METRICS)
    aw_map = dict(zip(hg['trial_id'], hg['one_minus_aw']))

    x_all, y_all = [], []
    rng = np.random.default_rng(42)

    for tid, grp in HG_TRIALS.items():
        if tid not in aw_map:
            continue
        d, t = tracked_tau50_profile(tid)
        sl, r, p = compute_slope(d, t)
        if sl is None:
            continue

        x_aw = aw_map[tid]
        x_all.append(x_aw)
        y_all.append(sl)
        jitter = rng.uniform(-0.006, 0.006)
        ax.scatter(x_aw + jitter, sl, c=COLORS[grp], s=35, alpha=0.65,
                   edgecolors='white', linewidths=0.4, zorder=3, label=grp)

    # De-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.3, borderpad=0.3)

    # Regression
    x_arr = np.array(x_all)
    y_arr = np.array(y_all)
    valid = ~np.isnan(y_arr)
    if valid.sum() >= 3:
        sl, ic, r, p, se = stats.linregress(x_arr[valid], y_arr[valid])
        r2 = r**2
        xfit = np.linspace(-0.02, 0.30, 100)
        ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)
        ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)
        print(f'  Panel K (all droplets): R²={r2:.3f}, slope={sl:.3f}, p={p:.2e}')

    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS, labelpad=3)
    ax.set_ylabel(r'd$\tau$/d$r$ (min/mm)', fontsize=LS, labelpad=3)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)
    ax.set_box_aspect(1)
    ax.text(-0.18, 1.05, 'J', transform=ax.transAxes,
            fontsize=PL, fontweight='bold', va='top')


def make_panels_JK():
    """Saves as panels_IJ to match original filename."""
    fig, (ax_j, ax_k) = plt.subplots(1, 2, figsize=(170 * MM, 75 * MM))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.17,
                        wspace=0.38)
    plot_panel_J(ax_j)
    plot_panel_K(ax_k)

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panels_IJ{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'Panels J+K → {FIG_OUT}/panels_IJ.*')


# --- Panel L: d2-law decomposition ---

def make_panel_L():
    hg = pd.read_csv(HG_METRICS)
    aw_map = dict(zip(hg['trial_id'], hg['one_minus_aw']))

    fig, ax = plt.subplots(figsize=(85 * MM, 85 * MM))
    fig.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.16)

    x_total, y_total = [], []
    x_size, y_size = [], []
    x_rate, y_rate = [], []
    rng = np.random.default_rng(42)

    for tid, grp in HG_TRIALS.items():
        if tid not in aw_map:
            continue
        total, size, rate = decompose_log_gradient(tid)
        if total is None:
            continue

        x_aw = aw_map[tid]
        x_total.append(x_aw); y_total.append(total)
        x_size.append(x_aw);  y_size.append(size)
        x_rate.append(x_aw);  y_rate.append(rate)

        j = rng.uniform(-0.004, 0.004)
        ax.scatter(x_aw + j, total, c=COLORS[grp], s=35, alpha=0.55,
                   edgecolors='white', linewidths=0.4, marker='o', zorder=4)
        ax.scatter(x_aw + j, size, c=COLORS[grp], s=35, alpha=0.55,
                   edgecolors='white', linewidths=0.4, marker='^', zorder=3)

    # Regression lines
    xfit = np.linspace(-0.02, 0.30, 100)
    for arr_x, arr_y, ls, lbl in [
        (x_total, y_total, '-',  'Total'),
        (x_size,  y_size,  '--', 'Size channel'),
    ]:
        xa, ya = np.array(arr_x), np.array(arr_y)
        if len(xa) >= 3:
            sl, ic, r, p, se = stats.linregress(xa, ya)
            ax.plot(xfit, ic + sl * xfit, color='k', ls=ls, lw=1.2,
                    alpha=0.7, zorder=2, label=f'{lbl} ($R^2$={r**2:.2f})')
            print(f'  Panel L {lbl}: R²={r**2:.3f}, slope={sl:.3f}, p={p:.2e}')

    # Shade the gap (= rate channel)
    xa_t, ya_t = np.array(x_total), np.array(y_total)
    xa_s, ya_s = np.array(x_size), np.array(y_size)
    if len(xa_t) >= 3 and len(xa_s) >= 3:
        sl_t, ic_t = stats.linregress(xa_t, ya_t)[:2]
        sl_s, ic_s = stats.linregress(xa_s, ya_s)[:2]
        ax.fill_between(xfit, ic_s + sl_s * xfit, ic_t + sl_t * xfit,
                        color='#C0392B', alpha=0.12, zorder=1,
                        label='Rate channel')

    ax.axhline(0, color='gray', ls=':', lw=0.5, alpha=0.5)
    ax.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
              handletextpad=0.4, borderpad=0.3)
    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS + 1, labelpad=4)
    ax.set_ylabel(r'd(ln $\tau_{50}$) / d$r$  (mm$^{-1}$)',
                  fontsize=LS, labelpad=4)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)

    for ext in ('.svg', '.pdf', '.png'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(FIG_OUT / f'panel_L{ext}',
                    bbox_inches='tight', facecolor='white', dpi=dpi)
    plt.close(fig)
    print(f'Panel L → {FIG_OUT}/panel_L.*')


# --- Main ---

def main():
    print('=== Generating manuscript panels (tracked) ===\n')
    make_panel_I()
    make_panels_JK()
    make_panel_L()
    print('\n=== Done ===')


if __name__ == '__main__':
    main()
