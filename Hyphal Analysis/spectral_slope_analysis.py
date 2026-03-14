#!/usr/bin/env python3
"""Spectral slope analysis comparing Aspergillus and Mucor hyphal surface texture."""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Paths
THIS_DIR   = Path(__file__).parent
OUTPUT_DIR = THIS_DIR

# Plot style
MM = 1 / 25.4
TS = 7.0; LS = 8.5; PL = 12.0
LW = 0.6

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':         TS,
    'axes.linewidth':    LW,
    'xtick.major.width': LW, 'ytick.major.width': LW,
    'xtick.major.size':  3.5, 'ytick.major.size':  3.5,
    'xtick.direction':   'out', 'ytick.direction':  'out',
    'svg.fonttype':      'none',
})

# Colors
C_ASP  = '#4CAF50'   # Green fungus (Aspergillus) — FFT images from Lab/Green/
C_MUC  = '#757575'   # White fungus (Mucor)       — FFT images from VMS_JPG/W2/

# Delta group means
# Green (Aspergillus): 279.5, 316.0, 297.8, 285.6, 311.7  → mean 298.1 ± 7.1
# White (Mucor):       198.5, 126.2, 120.0, 131.6, 123.0  → mean 139.9 ± 14.8
DELTA_ASP = np.array([279.5, 316.0, 297.8, 285.6, 311.7])
DELTA_MUC = np.array([198.5, 126.2, 120.0, 131.6, 123.0])

# Parameters
FREQ_LO = 0.01    # low-frequency cutoff for slope fit (cycles/px)
FREQ_HI = 0.45    # high-frequency cutoff (avoid aliasing/noise)


def load_spectra(genus):
    """Load full spectra CSV, return (freq_array, power_matrix[n_freq, n_transects])."""
    df = pd.read_csv(THIS_DIR / f'{genus}_full_spectra.csv')
    freq = df['frequency'].values
    transect_cols = [c for c in df.columns if c != 'frequency']
    power = df[transect_cols].values  # (n_freq, n_transects)
    return freq, power, transect_cols


def compute_slope(freq, power_col):
    """Fit log-log slope in [FREQ_LO, FREQ_HI] range. Return slope, r², n_pts."""
    valid = np.isfinite(power_col) & (power_col > 0) & \
            (freq >= FREQ_LO) & (freq <= FREQ_HI)
    if valid.sum() < 10:
        return np.nan, np.nan, 0

    lf = np.log10(freq[valid])
    lp = np.log10(power_col[valid])
    sl, ic, r, p, se = stats.linregress(lf, lp)
    return sl, r**2, int(valid.sum())


def main():
    # Load spectra
    freq_a, pow_a, cols_a = load_spectra('Aspergillus')
    freq_m, pow_m, cols_m = load_spectra('Mucor')
    n_asp = pow_a.shape[1]
    n_muc = pow_m.shape[1]
    print(f"Aspergillus: {n_asp} transects, {len(freq_a)} frequency bins")
    print(f"Mucor:       {n_muc} transects, {len(freq_m)} frequency bins")

    # Compute slopes
    slopes_asp = []
    r2_asp = []
    for j in range(n_asp):
        sl, r2, n = compute_slope(freq_a, pow_a[:, j])
        slopes_asp.append(sl)
        r2_asp.append(r2)

    slopes_muc = []
    r2_muc = []
    for j in range(n_muc):
        sl, r2, n = compute_slope(freq_m, pow_m[:, j])
        slopes_muc.append(sl)
        r2_muc.append(r2)

    slopes_asp = np.array(slopes_asp)
    slopes_muc = np.array(slopes_muc)
    r2_asp = np.array(r2_asp)
    r2_muc = np.array(r2_muc)

    # Remove any NaN slopes
    valid_a = np.isfinite(slopes_asp)
    valid_m = np.isfinite(slopes_muc)
    sa = slopes_asp[valid_a]
    sm = slopes_muc[valid_m]
    print(f"\nValid slopes: Aspergillus {len(sa)}/{n_asp}, Mucor {len(sm)}/{n_muc}")
    print(f"Total transects: {len(sa) + len(sm)}")

    # Statistics
    # Welch's t-test
    t_stat, t_p = stats.ttest_ind(sa, sm, equal_var=False)
    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(sa, sm, alternative='two-sided')
    # Cohen's d (pooled SD)
    n1, n2 = len(sa), len(sm)
    pooled_sd = np.sqrt(((n1-1)*sa.std(ddof=1)**2 + (n2-1)*sm.std(ddof=1)**2) / (n1+n2-2))
    cohens_d = (sa.mean() - sm.mean()) / pooled_sd
    # Effect size r from U
    r_eff = 1 - (2*u_stat) / (n1*n2)

    report = []
    report.append("=" * 65)
    report.append("SPECTRAL SLOPE ANALYSIS: Aspergillus vs Mucor")
    report.append("=" * 65)
    report.append(f"\nFrequency range for slope fit: [{FREQ_LO}, {FREQ_HI}] cycles/px")
    report.append(f"\nAspergillus (Green fungus — images from Lab/Green/):")
    report.append(f"  n = {n1} transects")
    report.append(f"  slope = {sa.mean():.3f} ± {sa.std(ddof=1):.3f} (mean ± SD)")
    report.append(f"  SEM = {sa.std(ddof=1)/np.sqrt(n1):.4f}")
    report.append(f"  median = {np.median(sa):.3f}")
    report.append(f"  range = [{sa.min():.3f}, {sa.max():.3f}]")
    report.append(f"  mean R² of log-log fit = {r2_asp[valid_a].mean():.3f}")
    report.append(f"\nMucor (White fungus — images from VMS_JPG/W2/):")
    report.append(f"  n = {n2} transects")
    report.append(f"  slope = {sm.mean():.3f} ± {sm.std(ddof=1):.3f} (mean ± SD)")
    report.append(f"  SEM = {sm.std(ddof=1)/np.sqrt(n2):.4f}")
    report.append(f"  median = {np.median(sm):.3f}")
    report.append(f"  range = [{sm.min():.3f}, {sm.max():.3f}]")
    report.append(f"  mean R² of log-log fit = {r2_muc[valid_m].mean():.3f}")
    report.append(f"\n--- Statistical Tests ---")
    report.append(f"Welch's t-test:  t = {t_stat:.3f}, p = {t_p:.3e}")
    report.append(f"Mann-Whitney U:  U = {u_stat:.1f}, p = {u_p:.3e}")
    report.append(f"Cohen's d = {cohens_d:.3f}")
    report.append(f"Effect size r = {r_eff:.3f}")
    report.append(f"\nShapiro-Wilk normality:")
    sw_a = stats.shapiro(sa)
    sw_m = stats.shapiro(sm)
    report.append(f"  Aspergillus: W = {sw_a.statistic:.4f}, p = {sw_a.pvalue:.4f}")
    report.append(f"  Mucor:       W = {sw_m.statistic:.4f}, p = {sw_m.pvalue:.4f}")

    # Levene's test for equal variances
    lev_stat, lev_p = stats.levene(sa, sm)
    report.append(f"\nLevene's test (equal var): F = {lev_stat:.3f}, p = {lev_p:.4f}")

    # KS test
    ks_stat, ks_p = stats.ks_2samp(sa, sm)
    report.append(f"Kolmogorov-Smirnov: D = {ks_stat:.3f}, p = {ks_p:.3e}")

    # δ values
    report.append(f"\n--- Dry-Zone Width (δ) from condensation data ---")
    report.append(f"Green (Aspergillus): δ = {DELTA_ASP.mean():.1f} ± {DELTA_ASP.std(ddof=1)/np.sqrt(5):.1f} µm (mean ± SEM, n=5)")
    report.append(f"White (Mucor):       δ = {DELTA_MUC.mean():.1f} ± {DELTA_MUC.std(ddof=1)/np.sqrt(5):.1f} µm (mean ± SEM, n=5)")

    report_txt = '\n'.join(report)
    print(report_txt)

    with open(OUTPUT_DIR / 'spectral_slope_stats.txt', 'w') as f:
        f.write(report_txt)

    # Save per-transect CSV
    rows = []
    for j, col in enumerate(cols_a):
        if valid_a[j]:
            rows.append({'transect_id': col, 'genus': 'Aspergillus',
                         'slope': slopes_asp[j], 'r2': r2_asp[j]})
    for j, col in enumerate(cols_m):
        if valid_m[j]:
            rows.append({'transect_id': col, 'genus': 'Mucor',
                         'slope': slopes_muc[j], 'r2': r2_muc[j]})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / 'spectral_slope_results.csv', index=False)

    # --- Figure ---
    fig = plt.figure(figsize=(180*MM, 60*MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 0.8],
                          left=0.08, right=0.97, top=0.88, bottom=0.18,
                          wspace=0.40)

    # Panel A: Mean power spectra
    ax_a = fig.add_subplot(gs[0])

    # Compute mean spectrum for each genus (geometric mean in log space)
    # Aspergillus
    mask_a = (freq_a >= 0.002) & (freq_a <= FREQ_HI)
    log_pow_a = np.log10(np.where(pow_a > 0, pow_a, np.nan))
    mean_log_a = np.nanmean(log_pow_a[mask_a], axis=1)
    sem_log_a = np.nanstd(log_pow_a[mask_a], axis=1, ddof=1) / np.sqrt(
        np.sum(np.isfinite(log_pow_a[mask_a]), axis=1))

    # Mucor
    mask_m = (freq_m >= 0.002) & (freq_m <= FREQ_HI)
    log_pow_m = np.log10(np.where(pow_m > 0, pow_m, np.nan))
    mean_log_m = np.nanmean(log_pow_m[mask_m], axis=1)
    sem_log_m = np.nanstd(log_pow_m[mask_m], axis=1, ddof=1) / np.sqrt(
        np.sum(np.isfinite(log_pow_m[mask_m]), axis=1))

    fa = freq_a[mask_a]
    fm = freq_m[mask_m]

    ax_a.plot(fa, 10**mean_log_a, color=C_ASP, lw=1.2, label='Aspergillus', zorder=3)
    ax_a.fill_between(fa, 10**(mean_log_a - sem_log_a), 10**(mean_log_a + sem_log_a),
                       color=C_ASP, alpha=0.15, zorder=2)

    ax_a.plot(fm, 10**mean_log_m, color=C_MUC, lw=1.2, label='Mucor', zorder=3)
    ax_a.fill_between(fm, 10**(mean_log_m - sem_log_m), 10**(mean_log_m + sem_log_m),
                       color=C_MUC, alpha=0.2, zorder=2)

    # Slope fit reference lines
    f_ref = np.array([FREQ_LO, FREQ_HI])
    # Aspergillus reference
    ic_a = np.nanmean(mean_log_a) - sa.mean() * np.nanmean(np.log10(fa))
    ax_a.plot(f_ref, 10**(ic_a + sa.mean()*np.log10(f_ref)),
              color=C_ASP, ls='--', lw=0.7, alpha=0.6)
    # Mucor reference
    ic_m = np.nanmean(mean_log_m) - sm.mean() * np.nanmean(np.log10(fm))
    ax_a.plot(f_ref, 10**(ic_m + sm.mean()*np.log10(f_ref)),
              color=C_MUC, ls='--', lw=0.7, alpha=0.6)

    # Vertical lines for fit range
    ax_a.axvline(FREQ_LO, color='gray', ls=':', lw=0.5, alpha=0.4)
    ax_a.axvline(FREQ_HI, color='gray', ls=':', lw=0.5, alpha=0.4)

    ax_a.set_xscale('log')
    ax_a.set_yscale('log')
    ax_a.set_xlabel('Spatial frequency (cycles/px)', fontsize=TS)
    ax_a.set_ylabel('Power', fontsize=TS)
    ax_a.legend(fontsize=TS-1, framealpha=0.9, loc='upper right',
                handletextpad=0.3, borderpad=0.3)

    # Annotate slopes on plot
    ax_a.text(0.05, 0.08,
              f'β = {sa.mean():.2f}',
              transform=ax_a.transAxes, fontsize=TS-0.5, color=C_ASP,
              fontweight='bold')
    ax_a.text(0.05, 0.18,
              f'β = {sm.mean():.2f}',
              transform=ax_a.transAxes, fontsize=TS-0.5, color=C_MUC,
              fontweight='bold')

    for sp in ['top', 'right']:
        ax_a.spines[sp].set_visible(False)

    ax_a.text(-0.14, 1.08, 'A', transform=ax_a.transAxes,
              fontsize=PL, fontweight='bold', va='top')

    # Panel B: Slope distributions
    ax_b = fig.add_subplot(gs[1])

    # Box + strip plot hybrid
    bp = ax_b.boxplot([sa, sm], positions=[1, 2], widths=0.5,
                       patch_artist=True, showfliers=False,
                       medianprops=dict(color='white', lw=1.2),
                       whiskerprops=dict(lw=0.8),
                       capprops=dict(lw=0.8))
    bp['boxes'][0].set_facecolor(C_ASP)
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(C_MUC)
    bp['boxes'][1].set_alpha(0.6)

    # Jittered points
    rng = np.random.default_rng(42)
    jitter_a = rng.uniform(-0.15, 0.15, len(sa))
    jitter_m = rng.uniform(-0.15, 0.15, len(sm))
    ax_b.scatter(1 + jitter_a, sa, s=8, c=C_ASP, alpha=0.5,
                 edgecolors='none', zorder=3)
    ax_b.scatter(2 + jitter_m, sm, s=8, c=C_MUC, alpha=0.5,
                 edgecolors='none', zorder=3)

    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(['Aspergillus\n(Green)', 'Mucor\n(White)'], fontsize=TS)
    ax_b.set_ylabel('Spectral slope β', fontsize=TS)

    # Significance bracket
    y_max = max(sa.max(), sm.max()) + 0.15
    ax_b.plot([1, 1, 2, 2], [y_max, y_max+0.08, y_max+0.08, y_max],
              color='black', lw=0.8)
    sig_str = f'p = {t_p:.1e}\nd = {abs(cohens_d):.2f}'
    ax_b.text(1.5, y_max + 0.10, sig_str, ha='center', va='bottom',
              fontsize=TS-1)

    for sp in ['top', 'right']:
        ax_b.spines[sp].set_visible(False)

    ax_b.text(-0.18, 1.08, 'B', transform=ax_b.transAxes,
              fontsize=PL, fontweight='bold', va='top')

    # Panel C: Slope vs delta
    ax_c = fig.add_subplot(gs[2])

    # Group means and SEMs
    slope_means = [sa.mean(), sm.mean()]
    slope_sems = [sa.std(ddof=1)/np.sqrt(len(sa)),
                  sm.std(ddof=1)/np.sqrt(len(sm))]
    delta_means = [DELTA_ASP.mean(), DELTA_MUC.mean()]
    delta_sems = [DELTA_ASP.std(ddof=1)/np.sqrt(5),
                  DELTA_MUC.std(ddof=1)/np.sqrt(5)]

    ax_c.errorbar(slope_means[0], delta_means[0],
                  xerr=slope_sems[0], yerr=delta_sems[0],
                  fmt='o', color=C_ASP, markersize=7,
                  markeredgecolor='white', markeredgewidth=0.5,
                  capsize=3, capthick=0.8, elinewidth=0.8,
                  label='Aspergillus', zorder=3)
    ax_c.errorbar(slope_means[1], delta_means[1],
                  xerr=slope_sems[1], yerr=delta_sems[1],
                  fmt='o', color=C_MUC, markersize=7,
                  markeredgecolor='white', markeredgewidth=0.5,
                  capsize=3, capthick=0.8, elinewidth=0.8,
                  label='Mucor', zorder=3)

    # Connecting line (suggestive, not a fit)
    ax_c.plot(slope_means, delta_means, color='gray', ls='--', lw=0.7,
              alpha=0.5, zorder=1)

    ax_c.set_xlabel('Spectral slope β', fontsize=TS)
    ax_c.set_ylabel('δ (µm)', fontsize=TS)
    ax_c.legend(fontsize=TS-1, framealpha=0.9, loc='best',
                handletextpad=0.3, borderpad=0.3)

    for sp in ['top', 'right']:
        ax_c.spines[sp].set_visible(False)

    ax_c.text(-0.22, 1.08, 'C', transform=ax_c.transAxes,
              fontsize=PL, fontweight='bold', va='top')

    # Save
    for ext in ('.png', '.pdf', '.svg'):
        kw = {'bbox_inches': 'tight', 'facecolor': 'white'}
        if ext == '.png':
            kw['dpi'] = 300
        fig.savefig(OUTPUT_DIR / f'spectral_slope_figure{ext}', **kw)
    plt.close(fig)
    print(f"\nSaved spectral_slope_figure.*")
    print(f"Saved spectral_slope_results.csv")
    print(f"Saved spectral_slope_stats.txt")


if __name__ == '__main__':
    main()
