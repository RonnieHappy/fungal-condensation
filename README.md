# Fungal Hyphae Reorganize Condensation Fields as Distributed Hygroscopic Sinks

Code and analysis pipeline for reproducing all quantitative figures and tables in the manuscript.

## Repository Structure

```
├── FigureSchematic/        # Figure 1: Schematic panels B, C (droplet count & mean radius)
│   └── code/
│       ├── step0_segment_droplets.py    # Cellpose cyto3 segmentation
│       ├── step1_batch_process.py       # EDT + raycast distance computation
│       ├── step2_beysens_profile.py     # Beysens scaling (β = 1/3, 1)
│       └── step3_figure_panels.py       # Generate panels B, C
│
├── FigureHGAggregate/      # Figure 2: Hydrogel aggregate panels E–L
│   └── code/
│       ├── step1_batch_process.py       # Batch segmentation + distance
│       ├── step2_compute_metrics.py     # Tanh fit, zone metrics, δ
│       ├── step4_panels_BC.py           # Panels F, G (radial profiles)
│       ├── step5_heatmap.py             # Panel C (heatmap, Figure 3)
│       ├── step6_panels_GH.py           # Panels E, H (size gradient)
│       └── test_tracking/
│           ├── track_droplets.py        # Forward-lifetime tracking
│           └── make_manuscript_panels.py # Panels I, J, K, L (survival)
│
├── FigureFungi/            # Figure 3: Fungal panels B, D, E
│   └── code/
│       ├── step1_batch_process.py       # Batch process fungal trials
│       ├── step2_compute_metrics_fungi.py # Fungal-specific metrics
│       ├── step3_panel_B_universal_Rstar.py # Panel B (universal R*)
│       └── step5_universal_panels.py    # Panels D, E (universal collapse)
│
├── FigureRSR/              # Figure 4: Rain-shadow-ridge panel B
│   └── code/
│       └── step1_figure_RSR.py          # RSR scatter plot
│
├── FigureTable/            # Supplementary metrics tables
│   └── code/
│       ├── step1_compute_all_metrics.py # Broken-stick + all metrics
│       ├── step3_generate_table.py      # Format metrics table
│       └── step4_universal_table.py     # Universal comparison table
│
├── Hyphal Analysis/        # Hyphal spacing FFT analysis
│   ├── hyphae_fft_density.py            # FFT transect analysis
│   └── spectral_slope_analysis.py       # Spectral slope statistics
│
├── Figures/                # Assembled publication figures (PDF + SVG)
├── methods.txt             # Complete parameter reference
└── FIGURE_PANEL_GUIDE.md   # Panel-by-panel reproduction guide
```

## System Requirements

### Operating System
Tested on macOS 14+ (Apple Silicon). Should work on any OS with Python 3.10+.

### Python Dependencies
| Package | Tested Version | Purpose |
|---------|---------------|---------|
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.11 | Curve fitting, EDT, statistics |
| matplotlib | ≥ 3.8 | Figure generation |
| pandas | ≥ 2.0 | Data handling |
| scikit-image | ≥ 0.21 | Morphological operations, regionprops |
| Pillow | ≥ 10.0 | Image I/O |
| cellpose | ≥ 3.0 | Droplet segmentation (cyto3 model) |
| lifelines | ≥ 0.27 | Kaplan–Meier survival analysis |

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy scipy matplotlib pandas scikit-image Pillow cellpose lifelines
```

Typical install time: ~3 minutes on a standard machine.

## Demo / Reproducing Figures

### Quick demo (figure generation from pre-computed outputs)

Each `Figure*/output/` directory contains pre-computed intermediate data (CSV files) and output figures (SVG/PDF/PNG). To regenerate figures from intermediate data:

```bash
# Figure 1 panels B, C
python FigureSchematic/code/step3_figure_panels.py

# Figure 2 panels F, G
python FigureHGAggregate/code/step4_panels_BC.py

# Figure 2 panels E, H
python FigureHGAggregate/code/step6_panels_GH.py

# Figure 2 panels I, J, K, L
python FigureHGAggregate/code/test_tracking/make_manuscript_panels.py

# Figure 3 panel C (heatmap)
python FigureHGAggregate/code/step5_heatmap.py

# Figure 3 panel B
python FigureFungi/code/step3_panel_B_universal_Rstar.py

# Figure 3 panels D, E
python FigureFungi/code/step5_universal_panels.py

# Figure 4 panel B
python FigureRSR/code/step1_figure_RSR.py
```

Typical run time: < 30 seconds per script on a standard machine.

### Full pipeline (from raw images)

The full pipeline starting from raw microscopy images requires the raw data hosted on OSF. Place raw data directories under each `Figure*/raw_data/` and run the `step0`/`step1` scripts first, then proceed with subsequent steps in numerical order.

Steps involving Cellpose segmentation (`step0_segment_droplets.py`) take ~2–5 minutes per trial on a machine with GPU; ~10–20 minutes on CPU only.

## Key Algorithms

1. **Droplet segmentation**: Cellpose cyto3 model (diameter = 90 px, flow_threshold = 0.2, cellprob_threshold = 1.0) with 3 px morphological erosion
2. **Distance computation**: Euclidean distance transform (EDT) from source boundary; raycast method (100 boundary samples) for dry-zone width δ
3. **Radial profile fitting**: Hyperbolic tangent model R(d) for size transition; broken-stick regression R(r') for size gradient
4. **Survival analysis**: Forward-lifetime Kaplan–Meier from seed time t = 15 min; coalescence events right-censored
5. **Beysens scaling**: log–log R(t) with β = 1/3 (diffusion-limited) and β = 1 (coalescence-dominated)

See `methods.txt` for the complete parameter reference.

## Raw Data

Raw microscopy images and NPY segmentation masks are hosted on OSF.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18968891.svg)](https://doi.org/10.5281/zenodo.18968891)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> [Author list]. Fungal Hyphae Reorganize Condensation Fields as Distributed Hygroscopic Sinks. *Nature Communications* (2025). https://doi.org/10.5281/zenodo.18968891
