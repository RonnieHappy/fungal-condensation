#!/usr/bin/env python3
"""Near-field vs far-field R(t) and delta vs water activity panels."""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
THIS_DIR    = Path(__file__).parent
PROJECT_DIR = THIS_DIR.parent
OUTPUT_DIR  = PROJECT_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HG_METRICS = OUTPUT_DIR / 'hydrogel_metrics.csv'
AGG_DIR    = PROJECT_DIR / 'raw_data' / 'aggregate_edt'

# Colors
COLORS = {'Agar': '#3A9E6F', '1:1': '#5B8FC9', '2:1': '#C0392B'}

# Exemplar trials
EXEMPLAR_AGAR = 'agar.3'
EXEMPLAR_NACL = '2to1.2'

# Analysis parameters
T_RANGE     = (5.0, 14.0)    # time range for Panel C-left
T_WINDOW    = (14.5, 15.5)   # window for density (Panel C-right)
MAX_DIST_MM = 2.5
DIST_BIN_MM = 0.15           # histogram bin width for density profile

# Plot style
MM = 1 / 25.4
TS = 7.0;  LS = 8.5;  PL = 12.0
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


def style_ax(ax):
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=TS)


# --- Panel B: scatter plots vs (1 - a_w) ---

def plot_panel_B(ax, x, y, colors_arr, ylabel, panel_label):
    """Scatter + OLS regression on individual hydrogel trials."""
    for group, color in COLORS.items():
        mask = (colors_arr == group) & ~np.isnan(y)
        if mask.any():
            ax.scatter(x[mask], y[mask], c=color, s=35, alpha=0.65,
                       zorder=3, edgecolors='white', linewidths=0.4,
                       label=group)

    valid = ~np.isnan(y)
    sl, ic, r, p, se = stats.linregress(x[valid], y[valid])
    r2 = r**2
    xfit = np.linspace(-0.02, 0.30, 100)
    ax.plot(xfit, ic + sl * xfit, 'k--', lw=1.2, alpha=0.7, zorder=2)

    ax.text(0.95, 0.08, f'$R^2$ = {r2:.3f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=TS)

    ax.set_xlabel(r'$(1 - a_w)$', fontsize=LS, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=LS, labelpad=3)
    ax.set_xlim(-0.03, 0.30)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    style_ax(ax)

    if panel_label:
        ax.text(-0.18, 1.05, panel_label, transform=ax.transAxes,
                fontsize=PL, fontweight='bold', va='top')

    return r2, sl, ic, p


# --- Panel C-left: R(t) near-field vs far-field ---

def plot_Rt_near_far(ax, trial_id, delta, r0, color, label_prefix):
    """
    Per-droplet median R(t) near vs far, relative to dry zone.
    near = delta < distance_um <= delta + NEAR_WIDTH
    far  = distance_um > r0
    Reads _edt_droplets.csv (per-droplet data).
    """
    NEAR_WIDTH = 500  # µm from dry zone edge
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    df = pd.read_csv(path)
    df = df[(df['radius_um'] > 0) &
            (df['time_min'] >= T_RANGE[0]) &
            (df['time_min'] <= T_RANGE[1])].copy()

    near = df[(df['distance_um'] > delta) & (df['distance_um'] <= delta + NEAR_WIDTH)]
    far  = df[df['distance_um'] > r0]

    near_med = near.groupby('time_min')['radius_um'].median()
    far_med  = far.groupby('time_min')['radius_um'].median()

    print(f'  {trial_id}: delta={delta:.0f}, near({delta:.0f}-{delta+NEAR_WIDTH:.0f})={len(near)}, '
          f'far(>{r0:.0f})={len(far)}')

    if len(far_med) > 0:
        ax.plot(far_med.index, far_med.values, 's-', color=color,
                ms=3.5, lw=1.2, label=f'{label_prefix} far',
                markeredgewidth=0.3, zorder=3)
    if len(near_med) > 0:
        ax.plot(near_med.index, near_med.values, 'o-', color=color,
                ms=3, lw=1.0, label=f'{label_prefix} near',
                markeredgewidth=0.3, zorder=3)

    # Shaded band
    common_t = sorted(set(near_med.index) & set(far_med.index))
    if len(common_t) > 2:
        t_arr = np.array(common_t)
        ax.fill_between(t_arr,
                        near_med.reindex(t_arr).values,
                        far_med.reindex(t_arr).values,
                        color=color, alpha=0.15, zorder=1)


# --- Panel C-right: normalized density vs distance ---

def plot_Rd_scatter(ax, trial_id, delta, color, label):
    """Individual droplets (distance, radius) with binned-mean R(d) overlay."""
    path = AGG_DIR / f'{trial_id}_edt_droplets.csv'
    df = pd.read_csv(path)
    df = df[(df['time_min'] >= T_WINDOW[0]) &
            (df['time_min'] <= T_WINDOW[1]) &
            (df['radius_um'] > 0)].copy()

    dist_mm = df['distance_um'].values / 1000.0
    radius  = df['radius_um'].values

    # Individual droplets
    ax.scatter(dist_mm, radius, s=1.5, c=color, alpha=0.15,
               edgecolors='none', rasterized=True, zorder=1)

    # Binned mean R(d)
    edges = np.arange(0, MAX_DIST_MM + DIST_BIN_MM, DIST_BIN_MM)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_idx = np.digitize(dist_mm, edges) - 1
    means = []
    first = True
    for i in range(len(centers)):
        in_bin_mask = bin_idx == i
        in_bin = radius[in_bin_mask]
        if len(in_bin) >= 5:
            x_val = centers[i]
            if first:
                # Snap first point to leftmost droplet in that bin
                x_val = dist_mm[in_bin_mask].min()
                first = False
            means.append((x_val, np.mean(in_bin)))
    if means:
        xm, ym = zip(*means)
        ax.plot(xm, ym, 'o-', color=color, ms=3.5, lw=1.3,
                markeredgecolor='white', markeredgewidth=0.4,
                label=label, zorder=3)

    # Delta annotation (only for NaCl — agar has no meaningful dry zone)
    delta_mm = delta / 1000.0
    if 'NaCl' in label:
        ax.axvline(delta_mm, color=color, ls=':', lw=1.5, alpha=0.8)
        ax.text(delta_mm, 12, r'$\delta$', color=color,
                fontsize=TS + 1, ha='center', va='bottom', fontweight='bold')


# --- Main ---

def main():
    hg = pd.read_csv(HG_METRICS)
    hg['group'] = hg['hydrogel_type'].map({
        'agar': 'Agar', '1:1': '1:1', '2:1': '2:1'})

    print(f"Loaded {len(hg)} hydrogel trials")
    print(hg[['trial_id', 'group', 'one_minus_aw', 'delta_um', 'alpha']].to_string())

    # Exemplar deltas and r0
    agar_row = hg[hg['trial_id'] == EXEMPLAR_AGAR].iloc[0]
    nacl_row = hg[hg['trial_id'] == EXEMPLAR_NACL].iloc[0]
    delta_agar = agar_row['delta_um']
    delta_nacl = nacl_row['delta_um']
    rbreak_agar = agar_row['r0_um']
    rbreak_nacl = nacl_row['r0_um']

    # Arrays
    x_aw    = hg['one_minus_aw'].values
    y_alpha = hg['alpha'].values
    y_delta = hg['delta_um'].values
    groups  = hg['group'].values

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(170 * MM, 150 * MM))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=0.08,
                        hspace=0.38, wspace=0.38)
    ax_bl, ax_br = axes[0]
    ax_cl, ax_cr = axes[1]

    # Panel F: delta vs (1 - a_w)
    r2_d, sl, ic, p = plot_panel_B(
        ax_bl, x_aw, y_delta, groups,
        r'$\delta$ ($\mu$m)', 'F')
    ax_bl.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
                 handletextpad=0.3, borderpad=0.3)
    print(f"\nPanel F (δ): R²={r2_d:.3f}, slope={sl:.1f}, p={p:.2e}")

    # Supplementary: alpha vs (1 - a_w)
    r2_a, sl, ic, p = plot_panel_B(
        ax_br, x_aw, y_alpha, groups,
        r'Steepness $\alpha$ ($\mu$m$^{-1}$)', '')
    ax_br.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
                 handletextpad=0.3, borderpad=0.3)
    print(f"Supplementary (alpha): R²={r2_a:.3f}, slope={sl:.6f}, p={p:.2e}")

    # Panel C-left: R(t) near/far
    print('\n── Panel C-left ──')
    plot_Rt_near_far(ax_cl, EXEMPLAR_NACL, delta_nacl, rbreak_nacl,
                     COLORS['2:1'], 'NaCl')
    plot_Rt_near_far(ax_cl, EXEMPLAR_AGAR, delta_agar, rbreak_agar,
                     COLORS['Agar'], 'Agar')

    ax_cl.set_xlabel('Time (min)', fontsize=LS, labelpad=3)
    ax_cl.set_ylabel(r'Mean radius $R$ ($\mu$m)', fontsize=LS, labelpad=3)
    ax_cl.set_xlim(T_RANGE[0] - 0.5, T_RANGE[1] + 0.5)
    ax_cl.set_ylim(8, 60)
    ax_cl.legend(fontsize=TS - 1, loc='upper left', framealpha=0.9,
                 ncol=1, handletextpad=0.3, borderpad=0.3, labelspacing=0.3)
    style_ax(ax_cl)
    ax_cl.text(-0.18, 1.05, 'G', transform=ax_cl.transAxes,
               fontsize=PL, fontweight='bold', va='top')

    # Panel C-right: R(d) scatter
    plot_Rd_scatter(ax_cr, EXEMPLAR_NACL, delta_nacl, COLORS['2:1'], 'NaCl')
    plot_Rd_scatter(ax_cr, EXEMPLAR_AGAR, delta_agar, COLORS['Agar'], 'Agar')

    ax_cr.set_xlabel('Distance from source (mm)', fontsize=LS, labelpad=3)
    ax_cr.set_ylabel(r'Radius $R$ ($\mu$m)', fontsize=LS, labelpad=3)
    ax_cr.set_xlim(0, MAX_DIST_MM)
    ax_cr.set_ylim(0, 70)
    ax_cr.legend(fontsize=TS - 0.5, loc='upper left', framealpha=0.9,
                 handletextpad=0.3, borderpad=0.3)
    style_ax(ax_cr)

    # Save
    for ext in ('.png', '.pdf', '.svg'):
        dpi = 300 if ext == '.png' else None
        fig.savefig(OUTPUT_DIR / f'panels_FG{ext}',
                    bbox_inches='tight', facecolor='white',
                    dpi=dpi if dpi else 'figure')
    plt.close(fig)
    print(f"\nSaved to {OUTPUT_DIR}/panels_FG.*")


if __name__ == '__main__':
    main()
