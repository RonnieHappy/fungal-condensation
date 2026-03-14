#!/usr/bin/env python3
"""Segment condensation droplets from microscopy images using Cellpose."""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import io as sk_io
from skimage.measure import regionprops
from skimage.morphology import erosion, disk
from cellpose import models, plot

VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

# Segmentation parameters
DEFAULT_DIAMETER = 90
DEFAULT_FLOW_THRESHOLD = 0.2
DEFAULT_CELLPROB_THRESHOLD = 1.0
DEFAULT_EROSION_RADIUS = 3


def segment_folder(input_folder, model, diameter, flow_threshold,
                   cellprob_threshold, erosion_radius):
    """Segment all images in a folder and save masks + overlays."""
    output_folder = f"{input_folder}_output"
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"  Warning: folder '{input_folder}' not found, skipping.")
        return None

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith(VALID_EXTENSIONS)
    ])

    if not image_files:
        print(f"  No images found in '{input_folder}', skipping.")
        return None

    print(f"  Found {len(image_files)} images")
    results = []

    for filename in image_files:
        img_path = os.path.join(input_folder, filename)
        print(f"    {filename} ...", end=" ", flush=True)

        img = sk_io.imread(img_path)

        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

        # Post-processing: erode each mask to create spacing between touching droplets
        if erosion_radius > 0:
            clean_masks = np.zeros_like(masks)
            for region in regionprops(masks):
                minr, minc, maxr, maxc = region.bbox
                eroded = erosion(region.image.astype(np.uint8), disk(erosion_radius)).astype(bool)
                clean_masks[minr:maxr, minc:maxc][eroded] = region.label
            masks = clean_masks

        count = masks.max()

        # Save masks
        base = os.path.splitext(filename)[0]
        np.save(os.path.join(output_folder, f"{base}_masks.npy"), masks)

        # Save overlay visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        if count > 0:
            ax.imshow(plot.mask_rgb(masks), alpha=0.5)
        ax.axis('off')
        ax.set_title(f"{filename} — {count} droplets")
        fig.savefig(os.path.join(output_folder, f"{base}_segmentation.png"),
                    bbox_inches='tight', dpi=150)
        plt.close(fig)

        results.append({'filename': filename, 'droplet_count': count})
        print(f"{count} droplets")

    # Save summary
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_folder, 'segmentation_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"  -> {csv_path}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Segment condensation droplets with Cellpose")
    parser.add_argument('--folders', nargs='+', default=['1', '2', '3', '4', '5', '6'],
                        help='Input folders to process (default: 1 2 3 4 5 6)')
    parser.add_argument('--diameter', type=int, default=DEFAULT_DIAMETER,
                        help=f'Cellpose diameter (default: {DEFAULT_DIAMETER})')
    parser.add_argument('--flow-threshold', type=float, default=DEFAULT_FLOW_THRESHOLD,
                        help=f'Flow threshold for cell separation (default: {DEFAULT_FLOW_THRESHOLD})')
    parser.add_argument('--cellprob-threshold', type=float, default=DEFAULT_CELLPROB_THRESHOLD,
                        help=f'Cell probability threshold (default: {DEFAULT_CELLPROB_THRESHOLD})')
    parser.add_argument('--erosion-radius', type=int, default=DEFAULT_EROSION_RADIUS,
                        help=f'Erosion radius in pixels (default: {DEFAULT_EROSION_RADIUS})')
    args = parser.parse_args()

    import torch
    use_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    model = models.CellposeModel(model_type='cyto3', gpu=use_gpu)
    print(f"Cellpose model loaded (GPU: {use_gpu})")
    print(f"Parameters: diameter={args.diameter}, flow={args.flow_threshold}, "
          f"cellprob={args.cellprob_threshold}, erosion={args.erosion_radius}")

    for folder in args.folders:
        print(f"\n--- {folder} ---")
        segment_folder(folder, model, args.diameter, args.flow_threshold,
                       args.cellprob_threshold, args.erosion_radius)

    print("\nDone.")


if __name__ == '__main__':
    main()
