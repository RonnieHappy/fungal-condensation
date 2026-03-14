#!/usr/bin/env python3
"""Extract droplet measurements from Cellpose NPY mask files."""

import numpy as np
import pandas as pd
import json
import re
import argparse
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path as MplPath

# Paths
THIS_DIR   = Path(__file__).parent
RAW_DATA   = THIS_DIR.parent / 'raw_data'
AGG_DIR    = RAW_DATA / 'aggregate_edt'
AGG_DIR.mkdir(parents=True, exist_ok=True)


# Trial configuration
# source_type: 'ellipse' for hydrogels, 'polygon' for fungi
TRIAL_CONFIG = {
    # Green fungi
    'Green.1':  {'folder': 'Green.1',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'Green.2':  {'folder': 'Green.2',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'Green.3':  {'folder': 'Green.3',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'Green.4':  {'folder': 'Green.4',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'Green.5':  {'folder': 'Green.5',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    # White fungi
    'white.1':  {'folder': 'white.1',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'white.2':  {'folder': 'white.2',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'white.3':  {'folder': 'white.3',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'white.4':  {'folder': 'white.4',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'white.5':  {'folder': 'white.5',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    # Black fungi
    'black.1':  {'folder': 'black.1',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'black.2':  {'folder': 'black.2',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'black.3':  {'folder': 'black.3',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'black.4':  {'folder': 'black.4',  'npy_subdir': 'Results', 'source_type': 'polygon'},
    'black.5':  {'folder': 'black.5',  'npy_subdir': 'Results', 'source_type': 'polygon'},
}

BIN_WIDTH_UM = 200
MIN_DROPLETS = 3


# Core functions

def parse_timestamp_seconds(filename: str) -> float | None:
    m = re.search(r'(\d+)m(\d+)s', Path(filename).stem)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    return None


def load_calibration(cal_path: Path) -> dict:
    with open(cal_path) as f:
        return json.load(f)


def build_edt_map_ellipse(shape, cal: dict):
    """Build EDT distance map from calibration ellipse (for hydrogels)."""
    ell = cal['source_ellipse']
    cx, cy = ell['center_px']
    a = ell['width_px'] / 2
    b = ell['height_px'] / 2
    angle = np.radians(ell.get('angle_deg', 0))
    rows, cols = shape
    y, x = np.ogrid[:rows, :cols]
    dx, dy = x - cx, y - cy
    ca, sa = np.cos(angle), np.sin(angle)
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1
    edt = distance_transform_edt(~mask).astype(np.float32)
    edt[mask] = 0
    return edt, mask


def build_edt_map_polygon(shape, cal: dict):
    """Build EDT distance map from calibration polygon (for fungi)."""
    src = cal['source_boundary']
    px, py = src['polygon_x'], src['polygon_y']
    verts = np.column_stack([px, py])
    path  = MplPath(verts)
    rows, cols = shape
    yg, xg = np.mgrid[:rows, :cols]
    pts = np.column_stack([xg.ravel(), yg.ravel()])
    mask = path.contains_points(pts).reshape(shape)
    edt = distance_transform_edt(~mask).astype(np.float32)
    edt[mask] = 0
    return edt, mask


def process_frame(npy_path: Path, time_sec: float, pixel_size: float,
                  edt_map: np.ndarray) -> list[dict]:
    """Extract per-droplet measurements from one frame."""
    masks = np.load(npy_path)
    time_min = time_sec / 60.0
    rows = []
    for lbl in np.unique(masks):
        if lbl == 0:
            continue
        obj = masks == lbl
        area_px = obj.sum()
        if area_px < 4:
            continue
        radius_um = np.sqrt(area_px / np.pi) * pixel_size
        ys, xs = np.where(obj)
        cx, cy = xs.mean(), ys.mean()
        dist_px = edt_map[int(cy), int(cx)]
        dist_um = dist_px * pixel_size
        rows.append({'time_min': time_min, 'radius_um': radius_um,
                     'distance_um': dist_um, 'cx': cx, 'cy': cy})
    return rows


def compute_binned_statistics(droplet_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean radius per (time, distance bin)."""
    max_dist = droplet_df['distance_um'].max()
    bins = np.arange(0, max_dist + BIN_WIDTH_UM, BIN_WIDTH_UM)
    droplet_df = droplet_df.copy()
    droplet_df['distance_bin_um'] = (
        pd.cut(droplet_df['distance_um'], bins=bins,
               labels=bins[:-1] + BIN_WIDTH_UM / 2).astype(float))
    grouped = droplet_df.groupby(['time_min', 'distance_bin_um'])['radius_um']
    binned = grouped.agg(['mean', 'std', 'count', 'sem']).reset_index()
    binned.columns = ['time_min', 'distance_bin_um',
                      'mean_radius_um', 'std_radius_um', 'n_droplets', 'sem_radius_um']
    return binned[binned['n_droplets'] >= MIN_DROPLETS]


# Per-trial processing

def process_trial(trial_id: str, cfg: dict):
    trial_dir = RAW_DATA / cfg['folder']
    cal_path  = trial_dir / 'calibration.json'
    npy_dir   = trial_dir / cfg['npy_subdir']

    if not cal_path.exists():
        print(f'  [SKIP] {trial_id}: calibration.json not found at {cal_path}')
        return
    if not npy_dir.exists():
        print(f'  [SKIP] {trial_id}: NPY folder not found at {npy_dir}')
        print(f'         Copy Results/*.npy from T7 to {npy_dir}/')
        return

    cal = load_calibration(cal_path)
    pixel_size = cal['scale']['pixel_size_um']

    npy_files = sorted(f for f in npy_dir.glob('*_masks.npy')
                       if not f.name.startswith('._'))
    if not npy_files:
        print(f'  [SKIP] {trial_id}: no *_masks.npy found in {npy_dir}')
        return

    # Load first frame to get shape
    first_mask = np.load(npy_files[0])
    shape = first_mask.shape

    # Build EDT map
    source_type = cfg.get('source_type', 'ellipse')
    if source_type == 'polygon':
        edt_map, source_mask = build_edt_map_polygon(shape, cal)
    else:
        edt_map, source_mask = build_edt_map_ellipse(shape, cal)

    print(f'  {trial_id}: {len(npy_files)} frames, shape={shape}, '
          f'max_dist={edt_map.max()*pixel_size:.0f} µm')

    # Process frames
    all_droplets = []
    for f in npy_files:
        t = parse_timestamp_seconds(f.name)
        if t is None:
            continue
        all_droplets.extend(process_frame(f, t, pixel_size, edt_map))

    if not all_droplets:
        print(f'  [WARN] {trial_id}: no droplets extracted')
        return

    df = pd.DataFrame(all_droplets)
    print(f'         {len(df):,} droplet measurements extracted')

    # Save droplets CSV
    df.to_csv(AGG_DIR / f'{trial_id}_edt_droplets.csv', index=False)

    # Save binned statistics
    binned = compute_binned_statistics(df)
    binned.to_csv(AGG_DIR / f'{trial_id}_edt_binned_statistics.csv', index=False)

    # Save boundary polygon (in µm)
    src = cal.get('source_boundary', {})
    if src.get('polygon_x'):
        poly = pd.DataFrame({'x': np.array(src['polygon_x']) * pixel_size,
                             'y': np.array(src['polygon_y']) * pixel_size})
        poly.to_csv(AGG_DIR / f'{trial_id}_boundary_polygon.csv', index=False)

    print(f'  ✓ saved to {AGG_DIR}/')


# Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', help='Process only this trial ID')
    args = parser.parse_args()

    trials = TRIAL_CONFIG
    if args.trial:
        if args.trial not in TRIAL_CONFIG:
            print(f'Trial "{args.trial}" not in TRIAL_CONFIG')
        else:
            trials = {args.trial: TRIAL_CONFIG[args.trial]}

    print(f'Processing {len(trials)} trial(s) → {AGG_DIR}')
    for tid, cfg in trials.items():
        print(f'\n--- {tid} ---')
        process_trial(tid, cfg)
    print('\nDone.')
