#!/usr/bin/env python3
"""
FFT-based hyphal density analyser – v3.

Select multiple image frames, draw transect lines, tune sensitivity to
detect fine hyphal structure, and export structured JSON/CSV directly
processable by downstream analysis scripts.

Sensitivity controls
--------------------
  The FFT often has large low-frequency power from gradual intensity
  gradients.  Two parameters let you focus on the fine structure:

    Low-freq cutoff (f/F)  – ignore frequencies below this threshold
                              (filters out slow gradients)
    Peak threshold   (t/T) – fraction of max power required for a peak
                              to be reported  (0→all, 1→only the max)

Controls
--------
  Click twice       define a transect line
  r                 reset current line
  n                 new transect (keep previous as history)
  h                 toggle help overlay
  s                 save current frame (CSV + PNG + JSON)
  S                 batch-save ALL analysed frames
  c                 set calibration (px → µm)
  w / W             decrease / increase band width
  f / F             decrease / increase low-freq cutoff
  t / T             decrease / increase peak threshold
  g                 toggle grayscale / colour
  left / right      navigate images
  q / Esc           quit

Usage
-----
  python3 hyphae_fft_density.py                        # multi-file dialog
  python3 hyphae_fft_density.py img1.tif img2.tif ...  # explicit files
  python3 hyphae_fft_density.py /path/to/folder        # all images in dir
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

# ── output directory ─────────────────────────────────────────────────

OUTPUT_DIR = Path.home() / "Downloads" / "OSF" / "FFT"

# ── colours ──────────────────────────────────────────────────────────

_TRANSECT_COLOURS = [
    "#00e5ff", "#76ff03", "#ff9100", "#e040fb",
    "#ffea00", "#00e676", "#ff5252", "#448aff",
]
_DOT_COLOUR   = "#ff1744"
_PEAK_COLOURS = ["#ff9800", "#4caf50", "#9c27b0", "#00bcd4", "#ff5722"]
_BG_DARK      = "#1e1e1e"
_BG_PANEL     = "#2a2a2a"
_FG_TEXT      = "#e0e0e0"
_FG_DIM       = "#888888"
_GRID_COLOUR  = "#444444"
_CUTOFF_CLR   = "#ff5252"
_THRESH_CLR   = "#ffeb3b"

# ── helpers ──────────────────────────────────────────────────────────

def _macos_multi_file_dialog():
    """macOS native multi-file picker. Returns list of paths."""
    script = (
        'set fList to choose file with prompt '
        '"Select image(s) – hold Cmd to select multiple" '
        'of type {"public.image"} with multiple selections allowed\n'
        'set out to ""\n'
        'repeat with f in fList\n'
        '  set out to out & POSIX path of f & linefeed\n'
        'end repeat\n'
        'return out'
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, check=True,
        )
        paths = [p.strip() for p in r.stdout.strip().split("\n") if p.strip()]
        return paths
    except subprocess.CalledProcessError:
        return []


def _macos_folder_dialog():
    script = (
        'set f to choose folder with prompt "Select image folder"\n'
        "return POSIX path of f"
    )
    try:
        r = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return None


IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def _collect_images_from_path(path: Path):
    if path.is_file():
        return [path]
    return sorted(
        p for p in path.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and not p.name.startswith("._")
    )


def _extract_profile(img_gray, p0, p1, n_samples=None):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if n_samples is None:
        n_samples = max(int(round(length)), 2)
    t = np.linspace(0.0, 1.0, n_samples)
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    h, w = img_gray.shape
    xi = np.clip(xs, 0, w - 1)
    yi = np.clip(ys, 0, h - 1)
    x0i = np.floor(xi).astype(int)
    y0i = np.floor(yi).astype(int)
    x1i = np.minimum(x0i + 1, w - 1)
    y1i = np.minimum(y0i + 1, h - 1)
    dx = xi - x0i
    dy = yi - y0i
    profile = (
        img_gray[y0i, x0i] * (1 - dx) * (1 - dy)
        + img_gray[y0i, x1i] * dx * (1 - dy)
        + img_gray[y1i, x0i] * (1 - dx) * dy
        + img_gray[y1i, x1i] * dx * dy
    )
    distances = t * length
    coords = np.column_stack([xs, ys])
    return distances, profile, coords


def _extract_band_profile(img_gray, p0, p1, band_width, n_samples=None):
    if band_width <= 1:
        d, prof, coords = _extract_profile(img_gray, p0, p1, n_samples)
        return d, prof, coords, prof.reshape(1, -1)
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if n_samples is None:
        n_samples = max(int(round(length)), 2)
    ddx, ddy = x1 - x0, y1 - y0
    norm = math.hypot(ddx, ddy)
    if norm < 1e-9:
        d, prof, coords = _extract_profile(img_gray, p0, p1, n_samples)
        return d, prof, coords, prof.reshape(1, -1)
    nx, ny = -ddy / norm, ddx / norm
    offsets = np.linspace(-band_width / 2, band_width / 2, int(band_width))
    profiles = []
    for off in offsets:
        pp0 = (x0 + nx * off, y0 + ny * off)
        pp1 = (x1 + nx * off, y1 + ny * off)
        _, prof, _ = _extract_profile(img_gray, pp0, pp1, n_samples)
        profiles.append(prof)
    strip = np.array(profiles)
    mean_prof = strip.mean(axis=0)
    t = np.linspace(0.0, 1.0, n_samples)
    distances = t * length
    coords_centre = np.column_stack([
        x0 + t * (x1 - x0), y0 + t * (y1 - y0),
    ])
    return distances, mean_prof, coords_centre, strip


def _compute_fft(profile, pixel_spacing=1.0):
    n = len(profile)
    centered = profile - profile.mean()
    window = np.hanning(n)
    windowed = centered * window
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=pixel_spacing)
    return freqs[1:], power[1:]


def _find_top_peaks(freqs, power, n_peaks=10, min_sep_bins=1,
                    freq_lo=0.0, threshold_frac=0.005):
    """Find top-N peaks above threshold and above low-freq cutoff.

    Uses local-maximum detection: a bin must be higher than both its
    immediate neighbours to qualify as a peak.  This avoids picking
    points on the slope of a broad peak.
    """
    mask = freqs >= freq_lo
    power_work = power.copy()
    power_work[~mask] = 0.0
    max_pow = power_work.max()
    if max_pow <= 0:
        return []
    thresh = max_pow * threshold_frac
    power_work[power_work < thresh] = 0.0

    # local-maximum filter
    n = len(power_work)
    is_peak = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if power_work[i] > 0:
            if power_work[i] >= power_work[i - 1] and power_work[i] >= power_work[i + 1]:
                is_peak[i] = True
    # edge cases
    if n > 0 and power_work[0] > 0 and (n == 1 or power_work[0] >= power_work[1]):
        is_peak[0] = True
    if n > 1 and power_work[-1] > 0 and power_work[-1] >= power_work[-2]:
        is_peak[-1] = True

    candidate_indices = np.where(is_peak)[0]
    if len(candidate_indices) == 0:
        return []

    # sort by power descending and pick with minimum separation
    order = candidate_indices[np.argsort(-power_work[candidate_indices])]
    peaks = []
    taken = set()
    for idx in order:
        if len(peaks) >= n_peaks:
            break
        if any(abs(idx - t) < min_sep_bins for t in taken):
            continue
        peaks.append(int(idx))
        taken.add(idx)
    return sorted(peaks)


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(_BG_PANEL)
    ax.set_title(title, fontsize=10, fontweight="bold", color=_FG_TEXT)
    ax.set_xlabel(xlabel, fontsize=8, color=_FG_DIM)
    ax.set_ylabel(ylabel, fontsize=8, color=_FG_DIM)
    ax.tick_params(colors=_FG_DIM, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(_GRID_COLOUR)
    ax.grid(True, alpha=0.25, color=_GRID_COLOUR)


# ── main application ─────────────────────────────────────────────────

class HyphaeFFTApp:
    def __init__(self, image_paths: list[Path], out_dir: Path):
        self.paths = image_paths
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.cal = 1.0
        self.cal_set = False
        self.show_gray = True
        self.band_width = 5
        self.clicks = []
        self.history = []

        # mode: "transect" (normal) or "calibrate" (ruler measurement)
        self.mode = "transect"
        self._cal_clicks = []
        self._cal_img_backup = None   # stash current image during cal

        # sensitivity – defaults tuned for fine hyphal structure
        self.freq_cutoff = 0.0     # auto-set after first analysis
        self.peak_thresh = 0.005   # 0.5% of max power = very sensitive
        self._auto_cutoff = True   # auto-set cutoff on first transect

        # live-preview state
        self._rubber_line = None
        self._crosshair_h = None
        self._crosshair_v = None
        self._help_visible = False
        self._help_artists = []

        # accumulate results across all frames for batch export
        self.all_results = {}  # keyed by image name

        # ── figure layout ────────────────────────────────────────────
        self.fig = plt.figure(figsize=(16, 9.5), facecolor=_BG_DARK)
        gs = gridspec.GridSpec(
            3, 2, figure=self.fig,
            width_ratios=[1.55, 1],
            height_ratios=[1.0, 0.35, 1.0],
            hspace=0.38, wspace=0.28,
            left=0.05, right=0.97, top=0.92, bottom=0.08,
        )
        self.ax_img = self.fig.add_subplot(gs[:, 0])
        self.ax_prof = self.fig.add_subplot(gs[0, 1])
        self.ax_strip = self.fig.add_subplot(gs[1, 1])
        self.ax_fft = self.fig.add_subplot(gs[2, 1])

        # header
        self.fig.text(
            0.50, 0.97, "Hyphal Density – FFT Analyser",
            fontsize=13, fontweight="bold", color=_FG_TEXT,
            ha="center", va="top",
        )
        self.subtitle_text = self.fig.text(
            0.50, 0.945, "", fontsize=9, color=_FG_DIM,
            ha="center", va="top",
        )

        # ── bottom button bar ────────────────────────────────────────
        btn_y, btn_h = 0.015, 0.035
        btn_specs = [
            (0.030, 0.040, "◀ Prev"),
            (0.075, 0.040, "Next ▶"),
            (0.120, 0.045, "Reset"),
            (0.170, 0.040, "Save"),
            (0.215, 0.050, "SaveAll"),
            (0.270, 0.055, "Calibrate"),
            (0.330, 0.040, "Bnd-"),
            (0.375, 0.040, "Bnd+"),
            (0.420, 0.050, "Cut-"),
            (0.475, 0.050, "Cut+"),
            (0.530, 0.040, "Thr-"),
            (0.575, 0.040, "Thr+"),
            (0.620, 0.040, "Help"),
        ]
        self.btns = {}
        for bx, bw, label in btn_specs:
            ax_btn = self.fig.add_axes([bx, btn_y, bw, btn_h])
            ax_btn.set_facecolor("#333333")
            btn = Button(ax_btn, label, color="#333333", hovercolor="#555555")
            btn.label.set_color(_FG_TEXT)
            btn.label.set_fontsize(7)
            self.btns[label] = btn

        self.btns["◀ Prev"].on_clicked(lambda _: self._go_prev())
        self.btns["Next ▶"].on_clicked(lambda _: self._go_next())
        self.btns["Reset"].on_clicked(lambda _: self._reset())
        self.btns["Save"].on_clicked(lambda _: self._save_current())
        self.btns["SaveAll"].on_clicked(lambda _: self._save_batch())
        self.btns["Calibrate"].on_clicked(lambda _: self._ask_calibration())
        self.btns["Bnd-"].on_clicked(lambda _: self._change_band(-2))
        self.btns["Bnd+"].on_clicked(lambda _: self._change_band(+2))
        self.btns["Cut-"].on_clicked(lambda _: self._change_cutoff(-1))
        self.btns["Cut+"].on_clicked(lambda _: self._change_cutoff(+1))
        self.btns["Thr-"].on_clicked(lambda _: self._change_thresh(-1))
        self.btns["Thr+"].on_clicked(lambda _: self._change_thresh(+1))
        self.btns["Help"].on_clicked(lambda _: self._toggle_help())

        # status bar
        self.status_text = self.fig.text(
            0.74, 0.022, "", fontsize=8, color=_FG_DIM,
            ha="left", va="bottom", family="monospace",
        )
        self.param_text = self.fig.text(
            0.97, 0.022, "", fontsize=7.5, color="#80cbc4",
            ha="right", va="bottom", family="monospace",
        )

        # press tracking for drag detection
        self._press_xy = None

        # events
        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._load_image()
        self._update_param_label()
        plt.show()

        # no auto-save — only save when user explicitly presses s/S

    # ── image I/O ────────────────────────────────────────────────────

    def _load_image(self):
        path = self.paths[self.idx]
        img = Image.open(path)
        self.img_rgb = np.array(img.convert("RGB"))
        self.img_gray = np.array(img.convert("L")).astype(float)
        self.clicks.clear()
        self.history.clear()
        self._draw_image()
        self._clear_right_panels()
        self._update_subtitle()

    def _draw_image(self):
        self.ax_img.clear()
        self.ax_img.set_facecolor(_BG_DARK)
        if self.show_gray:
            self.ax_img.imshow(self.img_gray, cmap="gray", aspect="equal")
        else:
            self.ax_img.imshow(self.img_rgb, aspect="equal")
        self.ax_img.set_title(
            self.paths[self.idx].name,
            fontsize=10, fontweight="bold", color=_FG_TEXT,
        )
        self.ax_img.tick_params(colors=_FG_DIM, labelsize=7)
        for spine in self.ax_img.spines.values():
            spine.set_color(_GRID_COLOUR)
        for i, h in enumerate(self.history):
            c = _TRANSECT_COLOURS[i % len(_TRANSECT_COLOURS)]
            self.ax_img.plot(
                [h["p0"][0], h["p1"][0]], [h["p0"][1], h["p1"][1]],
                "-", color=c, linewidth=1.2, alpha=0.5,
            )
            mx = (h["p0"][0] + h["p1"][0]) / 2
            my = (h["p0"][1] + h["p1"][1]) / 2
            self.ax_img.text(
                mx, my, f"#{i+1}", fontsize=7, color=c,
                ha="center", va="bottom", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=_BG_DARK, ec=c, alpha=0.7),
            )
        self._rubber_line = None
        self._crosshair_h = None
        self._crosshair_v = None
        self.fig.canvas.draw_idle()

    def _clear_right_panels(self):
        for ax in (self.ax_prof, self.ax_strip, self.ax_fft):
            ax.clear()
            ax.set_facecolor(_BG_PANEL)
            ax.tick_params(colors=_FG_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(_GRID_COLOUR)
        self.ax_prof.text(
            0.5, 0.5, "Intensity Profile",
            transform=self.ax_prof.transAxes, ha="center", va="center",
            fontsize=11, color=_FG_DIM, alpha=0.4,
        )
        self.ax_strip.text(
            0.5, 0.5, "ROI Strip Preview",
            transform=self.ax_strip.transAxes, ha="center", va="center",
            fontsize=11, color=_FG_DIM, alpha=0.4,
        )
        self.ax_fft.text(
            0.5, 0.5, "FFT Power Spectrum",
            transform=self.ax_fft.transAxes, ha="center", va="center",
            fontsize=11, color=_FG_DIM, alpha=0.4,
        )
        self.fig.canvas.draw_idle()

    # ── interaction ──────────────────────────────────────────────────
    #
    #   Supports BOTH workflows:
    #     Drag:        press A → drag → release B  (line A→B)
    #     Click-click: press A (release same spot) → press B  (line A→B)
    #
    #   If a transect already exists, the next press auto-stores it.

    def _on_press(self, event):
        if event.inaxes != self.ax_img or event.button != 1:
            return
        self._press_xy = (event.xdata, event.ydata)

        if self.mode == "calibrate":
            if len(self._cal_clicks) == 0:
                self._cal_clicks = [(event.xdata, event.ydata)]
                self.ax_img.plot(
                    event.xdata, event.ydata, "s", color="#ffea00",
                    markersize=10, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=15,
                )
                self._update_status("CALIBRATE: drag to other end, or click it")
                self.fig.canvas.draw_idle()
            elif len(self._cal_clicks) == 1:
                # second click → complete calibration
                self._complete_calibration((event.xdata, event.ydata))
            return

        # auto-store previous transect if complete
        if len(self.clicks) >= 2:
            self._new_transect()

        if len(self.clicks) == 0:
            # first point
            self.clicks = [(event.xdata, event.ydata)]
            self.ax_img.plot(
                event.xdata, event.ydata, "o", color=_DOT_COLOUR,
                markersize=7, markeredgecolor="white",
                markeredgewidth=1.0, zorder=10,
            )
            self._update_status("Drag to endpoint, or click it")
            self.fig.canvas.draw_idle()
        elif len(self.clicks) == 1:
            # second click (click-click mode) → complete transect
            self._complete_transect((event.xdata, event.ydata))

    def _on_release(self, event):
        if event.inaxes != self.ax_img or self._press_xy is None:
            return
        release_xy = (event.xdata, event.ydata)
        press_xy = self._press_xy
        self._press_xy = None
        drag_dist = math.hypot(release_xy[0] - press_xy[0],
                               release_xy[1] - press_xy[1])
        was_drag = drag_dist > 15  # > 15 px = intentional drag

        # ── calibration ──────────────────────────────────────────────
        if self.mode == "calibrate":
            if len(self._cal_clicks) == 1:
                if was_drag:
                    # drag completed → use release as second point
                    self._complete_calibration(release_xy)
                # else: just a click, wait for second click via next press
                elif len(self._cal_clicks) == 1:
                    pass  # wait for second click
            return

        # ── transect ─────────────────────────────────────────────────
        if was_drag and len(self.clicks) == 1:
            # drag completed → use release point as endpoint
            self._complete_transect(release_xy)
        elif not was_drag and len(self.clicks) == 1:
            # just a click (no drag) — this IS the first point, wait
            pass

    def _on_click_second(self, event):
        """Called on press when we already have 1 click (click-click mode)."""
        pass  # handled in _on_press

    def _complete_transect(self, end_xy):
        """Finalize a transect line from clicks[0] to end_xy."""
        if self._rubber_line is not None:
            self._rubber_line.remove()
            self._rubber_line = None

        self.clicks.append(end_xy)
        p0, p1 = self.clicks
        cidx = len(self.history) % len(_TRANSECT_COLOURS)

        self.ax_img.plot(
            end_xy[0], end_xy[1], "o", color=_DOT_COLOUR, markersize=7,
            markeredgecolor="white", markeredgewidth=1.0, zorder=10,
        )
        self.ax_img.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            "-", color=_TRANSECT_COLOURS[cidx], linewidth=2.0, zorder=5,
        )
        if self.band_width > 1:
            self._draw_band_edges(p0, p1, _TRANSECT_COLOURS[cidx])
        self._analyse()
        self.fig.canvas.draw_idle()

    def _complete_calibration(self, end_xy):
        """Finalize ruler measurement."""
        start = self._cal_clicks[0]
        self._cal_clicks.append(end_xy)

        self.ax_img.plot(
            end_xy[0], end_xy[1], "s", color="#ffea00", markersize=10,
            markeredgecolor="white", markeredgewidth=1.5, zorder=15,
        )
        self.ax_img.plot(
            [start[0], end_xy[0]], [start[1], end_xy[1]],
            "-", color="#ffea00", linewidth=2.5, zorder=14,
        )
        if self._rubber_line is not None:
            self._rubber_line.remove()
            self._rubber_line = None
        self.fig.canvas.draw_idle()

        ruler_px = math.hypot(end_xy[0] - start[0], end_xy[1] - start[1])
        script = (
            'set ans to display dialog '
            '"Ruler distance in the image is '
            f'{ruler_px:.0f} px.\\n\\n'
            'Enter the real distance (mm):" '
            'default answer "1.0" '
            'with title "Ruler Calibration"\n'
            'return text returned of ans'
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True,
            )
            mm_val = float(r.stdout.strip())
            if mm_val > 0:
                um_per_px = (mm_val * 1000.0) / ruler_px
                self.cal = um_per_px
                self.cal_set = True
                self._update_status(
                    f"Calibrated: {um_per_px:.4f} µm/px  "
                    f"({ruler_px:.0f} px = {mm_val} mm)"
                )
        except (subprocess.CalledProcessError, ValueError):
            self._update_status("Calibration cancelled")
        self._finish_calibration()
        self.fig.canvas.draw_idle()

    def _draw_band_edges(self, p0, p1, colour):
        ddx, ddy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(ddx, ddy)
        if norm < 1e-9:
            return
        nx, ny = -ddy / norm, ddx / norm
        hw = self.band_width / 2
        for sign in (-1, 1):
            self.ax_img.plot(
                [p0[0] + nx * hw * sign, p1[0] + nx * hw * sign],
                [p0[1] + ny * hw * sign, p1[1] + ny * hw * sign],
                "--", color=colour, linewidth=0.8, alpha=0.45, zorder=4,
            )

    def _on_motion(self, event):
        if event.inaxes != self.ax_img:
            if self._crosshair_h is not None:
                self._crosshair_h.set_visible(False)
                self._crosshair_v.set_visible(False)
                self.fig.canvas.draw_idle()
            return
        if self._crosshair_h is None:
            self._crosshair_h = self.ax_img.axhline(
                event.ydata, color=_FG_DIM, lw=0.5, alpha=0.4, zorder=1,
            )
            self._crosshair_v = self.ax_img.axvline(
                event.xdata, color=_FG_DIM, lw=0.5, alpha=0.4, zorder=1,
            )
        else:
            self._crosshair_h.set_ydata([event.ydata])
            self._crosshair_h.set_visible(True)
            self._crosshair_v.set_xdata([event.xdata])
            self._crosshair_v.set_visible(True)
        # rubber-band line after first click (follows mouse)
        active_clicks = (self._cal_clicks if self.mode == "calibrate"
                         else self.clicks)
        rb_colour = "#ffea00" if self.mode == "calibrate" else "#ffffff"
        if len(active_clicks) == 1:
            px, py = active_clicks[0]
            if self._rubber_line is None:
                self._rubber_line, = self.ax_img.plot(
                    [px, event.xdata], [py, event.ydata],
                    "--", color=rb_colour, linewidth=1.5, alpha=0.7, zorder=8,
                )
            else:
                self._rubber_line.set_data(
                    [px, event.xdata], [py, event.ydata],
                )
            dist = math.hypot(event.xdata - px, event.ydata - py)
            if self.mode == "calibrate":
                self._update_status(f"CALIBRATE: ruler = {dist:.0f} px")
            else:
                unit = "µm" if self.cal_set else "px"
                phys = dist * self.cal if self.cal_set else dist
                self._update_status(f"Length: {phys:.0f} {unit}")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        k = event.key
        if k == "r":
            self._reset()
        elif k == "z":
            self._undo()
        elif k == "n":
            self._new_transect()
        elif k == "s":
            self._save_current()
        elif k == "S":
            self._save_batch()
        elif k == "x":
            self._save_session()
        elif k == "c":
            self._ask_calibration()
        elif k == "h":
            self._toggle_help()
        elif k == "g":
            self.show_gray = not self.show_gray
            self._draw_image()
            if len(self.clicks) == 2:
                self._redraw_current_line()
        elif k == "w":
            self._change_band(-2)
        elif k == "W":
            self._change_band(+2)
        elif k == "f":
            self._change_cutoff(-1)
        elif k == "F":
            self._change_cutoff(+1)
        elif k == "t":
            self._change_thresh(-1)
        elif k == "T":
            self._change_thresh(+1)
        elif k == "right" and len(self.paths) > 1:
            self._go_next()
        elif k == "left" and len(self.paths) > 1:
            self._go_prev()
        elif k in ("q", "escape"):
            self._store_if_analysed()
            plt.close(self.fig)

    # ── transect management ──────────────────────────────────────────

    def _reset(self):
        """Reset only the CURRENT transect line. History is preserved."""
        # stash for undo
        self._undo_clicks = self.clicks.copy()
        self._undo_result = getattr(self, "_last_result", None)

        self.clicks.clear()
        if hasattr(self, "_last_result"):
            del self._last_result
        self._draw_image()       # redraws history transects
        self._clear_right_panels()
        self._update_status("Current line reset (z to undo) – history kept")

    def _undo(self):
        """Undo the last reset."""
        if not hasattr(self, "_undo_clicks") or not self._undo_clicks:
            self._update_status("Nothing to undo")
            return
        self.clicks = self._undo_clicks
        self._undo_clicks = []
        if self._undo_result is not None:
            self._last_result = self._undo_result
            self._undo_result = None
        self._draw_image()
        if len(self.clicks) == 2:
            self._redraw_current_line()
            self._analyse()
        self._update_status("Undone")

    def _new_transect(self):
        if len(self.clicks) < 2 or not hasattr(self, "_last_result"):
            return
        res = self._last_result
        self.history.append(dict(
            p0=res["p0"], p1=res["p1"],
            peaks=res["peaks"],
            unit=res["unit"],
        ))
        self._store_if_analysed()
        self.clicks.clear()
        self._draw_image()
        self._clear_right_panels()
        self._update_status(f"Transect #{len(self.history)} stored – draw next")

    def _go_next(self):
        if len(self.paths) <= 1:
            return
        self._store_if_analysed()
        self.idx = (self.idx + 1) % len(self.paths)
        self._load_image()

    def _go_prev(self):
        if len(self.paths) <= 1:
            return
        self._store_if_analysed()
        self.idx = (self.idx - 1) % len(self.paths)
        self._load_image()

    def _redraw_current_line(self):
        if len(self.clicks) == 2:
            p0, p1 = self.clicks
            cidx = len(self.history) % len(_TRANSECT_COLOURS)
            c = _TRANSECT_COLOURS[cidx]
            self.ax_img.plot(
                [p0[0], p1[0]], [p0[1], p1[1]],
                "-", color=c, linewidth=2.0, zorder=5,
            )
            if self.band_width > 1:
                self._draw_band_edges(p0, p1, c)
            for x, y in self.clicks:
                self.ax_img.plot(
                    x, y, "o", color=_DOT_COLOUR, markersize=7,
                    markeredgecolor="white", markeredgewidth=1.0, zorder=10,
                )
            self.fig.canvas.draw_idle()

    # ── parameter adjustments ────────────────────────────────────────

    def _change_band(self, delta):
        self.band_width = max(1, self.band_width + delta)
        self._update_param_label()
        if len(self.clicks) == 2:
            self._draw_image()
            self._redraw_current_line()
            self._analyse()

    def _change_cutoff(self, direction):
        """Shift low-freq cutoff.  Steps are adaptive to the frequency range."""
        self._auto_cutoff = False
        if hasattr(self, "_last_result") and len(self._last_result["freqs"]) > 0:
            fmax = self._last_result["freqs"][-1]
            step = fmax * 0.01      # 1% of max frequency per click (fine steps)
        else:
            step = 0.002
        self.freq_cutoff = max(0.0, self.freq_cutoff + direction * step)
        self._update_param_label()
        if len(self.clicks) == 2:
            self._analyse()

    def _change_thresh(self, direction):
        step = 0.02
        self.peak_thresh = max(0.0, min(0.98, self.peak_thresh + direction * step))
        self._update_param_label()
        if len(self.clicks) == 2:
            self._analyse()

    # ── FFT analysis ─────────────────────────────────────────────────

    def _analyse(self):
        p0, p1 = self.clicks
        line_len_px = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        sample_spacing = self.cal
        unit = "µm" if self.cal_set else "px"

        distances, profile, coords, strip = _extract_band_profile(
            self.img_gray, p0, p1, self.band_width,
        )
        distances_phys = distances * self.cal
        freqs, power = _compute_fft(profile, pixel_spacing=sample_spacing)

        # auto-set cutoff on first analysis: 10% of Nyquist frequency
        # this kills the dominant low-freq gradient peaks
        if self._auto_cutoff and len(freqs) > 0:
            nyquist = freqs[-1]
            self.freq_cutoff = nyquist * 0.10
            self._auto_cutoff = False
            self._update_param_label()

        peak_indices = _find_top_peaks(
            freqs, power, n_peaks=10, min_sep_bins=1,
            freq_lo=self.freq_cutoff, threshold_frac=self.peak_thresh,
        )
        peaks = []
        for pi in peak_indices:
            f = freqs[pi]
            wl = 1.0 / f if f > 0 else np.inf
            peaks.append(dict(idx=pi, freq=f, wavelength=wl, power=float(power[pi])))

        # ── intensity profile ────────────────────────────────────────
        self.ax_prof.clear()
        _style_ax(self.ax_prof, "Intensity Profile",
                  f"Distance ({unit})", "Intensity")
        self.ax_prof.fill_between(distances_phys, profile,
                                  color="#1976d2", alpha=0.15)
        self.ax_prof.plot(distances_phys, profile, color="#42a5f5", lw=1.0)
        for i, pk in enumerate(peaks[:3]):
            if pk["wavelength"] < distances_phys[-1]:
                positions = np.arange(0, distances_phys[-1], pk["wavelength"])
                for xp in positions:
                    self.ax_prof.axvline(
                        xp, color=_PEAK_COLOURS[i % len(_PEAK_COLOURS)],
                        lw=0.6, alpha=0.25, zorder=0,
                    )

        # ── ROI strip preview ────────────────────────────────────────
        self.ax_strip.clear()
        _style_ax(self.ax_strip, f"ROI Strip  (band={self.band_width} px)")
        display_strip = strip
        if strip.shape[0] < 3:
            display_strip = np.repeat(strip, max(1, 12 // strip.shape[0]), axis=0)
        self.ax_strip.imshow(
            display_strip, cmap="gray", aspect="auto",
            extent=[distances_phys[0], distances_phys[-1],
                    -self.band_width / 2, self.band_width / 2],
        )
        self.ax_strip.set_xlabel(f"Distance ({unit})", fontsize=8, color=_FG_DIM)
        self.ax_strip.set_ylabel("Offset (px)", fontsize=8, color=_FG_DIM)

        # ── FFT power spectrum ───────────────────────────────────────
        self.ax_fft.clear()
        _style_ax(self.ax_fft, "FFT Power Spectrum",
                  f"Spatial frequency (cycles/{unit})", "Power")
        self.ax_fft.fill_between(freqs, power, color="#d32f2f", alpha=0.12)
        self.ax_fft.plot(freqs, power, color="#ef5350", lw=1.0)

        # show low-freq cutoff zone
        if self.freq_cutoff > 0:
            self.ax_fft.axvspan(
                0, self.freq_cutoff,
                color=_CUTOFF_CLR, alpha=0.12, zorder=0,
            )
            self.ax_fft.axvline(
                self.freq_cutoff, color=_CUTOFF_CLR, lw=1.0,
                ls=":", alpha=0.7, zorder=1,
            )
            self.ax_fft.text(
                self.freq_cutoff, self.ax_fft.get_ylim()[1] * 0.98,
                " cutoff", fontsize=6, color=_CUTOFF_CLR,
                va="top", ha="left",
            )

        # show threshold line
        if len(power) > 0:
            valid_mask = freqs >= self.freq_cutoff
            max_pow_valid = power[valid_mask].max() if valid_mask.any() else power.max()
            thresh_line = max_pow_valid * self.peak_thresh
            self.ax_fft.axhline(
                thresh_line, color=_THRESH_CLR, lw=0.8,
                ls="--", alpha=0.5, zorder=1,
            )
            self.ax_fft.text(
                freqs[-1] * 0.01, thresh_line,
                f" threshold {self.peak_thresh:.0%}",
                fontsize=6, color=_THRESH_CLR, va="bottom",
            )

        # annotate peaks
        for i, pk in enumerate(peaks[:5]):
            colour = _PEAK_COLOURS[i % len(_PEAK_COLOURS)]
            self.ax_fft.axvline(pk["freq"], ls="--", color=colour, lw=1.2, alpha=0.7)
            label = (
                f"#{i+1} f={pk['freq']:.4f}\n"
                f"   λ={pk['wavelength']:.1f} {unit}"
            )
            y_anchor = 1.0 - i * 0.16
            self.ax_fft.annotate(
                label,
                xy=(pk["freq"], pk["power"]),
                xytext=(0.97, y_anchor), textcoords="axes fraction",
                ha="right", va="top", fontsize=7, color=colour,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc=_BG_DARK, ec=colour, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=colour, lw=0.8),
            )

        # summary box on image
        if peaks:
            pk1 = peaks[0]
            summary = (
                f"#1 density: {pk1['freq']:.4f} cyc/{unit}\n"
                f"   spacing: {pk1['wavelength']:.1f} {unit}\n"
                f"Line: {line_len_px:.0f} px  Band: {self.band_width} px\n"
                f"Cutoff: {self.freq_cutoff:.4f}  Thr: {self.peak_thresh:.0%}"
            )
            self.ax_img.text(
                0.02, 0.02, summary,
                transform=self.ax_img.transAxes, fontsize=7.5,
                fontfamily="monospace", color="#b2ff59",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc=_BG_DARK,
                          ec="#b2ff59", alpha=0.85),
                zorder=20,
            )

        # store
        self._last_result = dict(
            path=self.paths[self.idx],
            p0=p0, p1=p1,
            line_length_px=line_len_px,
            band_width=self.band_width,
            freq_cutoff=self.freq_cutoff,
            peak_thresh=self.peak_thresh,
            cal=self.cal, unit=unit,
            distances=distances_phys,
            profile=profile,
            freqs=freqs, power=power,
            peaks=peaks,
        )
        self._update_status(
            f"Peak: {peaks[0]['freq']:.4f} cyc/{unit}  "
            f"λ={peaks[0]['wavelength']:.1f} {unit}" if peaks else "No peaks found"
        )
        self.fig.canvas.draw_idle()

    # ── store result into batch accumulator ──────────────────────────

    def _store_if_analysed(self):
        if not hasattr(self, "_last_result"):
            return
        res = self._last_result
        name = res["path"].name
        if name not in self.all_results:
            self.all_results[name] = []
        # include current + history
        entry = self._result_to_dict(res)
        # check if identical entry already stored (avoid duplicates)
        if not any(e["x0"] == entry["x0"] and e["y0"] == entry["y0"]
                   and e["x1"] == entry["x1"] and e["y1"] == entry["y1"]
                   for e in self.all_results[name]):
            self.all_results[name].append(entry)

    def _result_to_dict(self, res):
        """Convert a result dict to a clean serialisable dict."""
        return {
            "image": res["path"].name,
            "image_path": str(res["path"]),
            "x0": round(res["p0"][0], 1),
            "y0": round(res["p0"][1], 1),
            "x1": round(res["p1"][0], 1),
            "y1": round(res["p1"][1], 1),
            "line_length_px": round(res["line_length_px"], 1),
            "band_width_px": res["band_width"],
            "calibration_um_per_px": res["cal"],
            "unit": res["unit"],
            "freq_cutoff": round(res["freq_cutoff"], 6),
            "peak_threshold": round(res["peak_thresh"], 4),
            "peaks": [
                {
                    "rank": i + 1,
                    "frequency": round(pk["freq"], 6),
                    "wavelength": round(pk["wavelength"], 2),
                    "power": round(pk["power"], 4),
                    "density_per_unit": round(pk["freq"], 6),
                }
                for i, pk in enumerate(res["peaks"])
            ],
            "profile_mean_intensity": round(float(res["profile"].mean()), 2),
            "profile_std_intensity": round(float(res["profile"].std()), 2),
            "profile_length_samples": len(res["profile"]),
            "spectrum_freqs": [round(float(f), 8) for f in res["freqs"]],
            "spectrum_power": [round(float(p), 4) for p in res["power"]],
        }

    # ── calibration ──────────────────────────────────────────────────

    def _ask_calibration(self):
        """Enter calibration mode: pick a ruler image, click two points,
        enter the known distance in mm."""
        # open file dialog to select the ruler / scale bar image
        script = (
            'set f to choose file with prompt '
            '"Select an image with a ruler or scale bar" '
            'of type {"public.image"}\nreturn POSIX path of f'
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, check=True,
            )
            cal_path = r.stdout.strip()
        except subprocess.CalledProcessError:
            self._update_status("Calibration cancelled")
            return

        if not cal_path:
            return

        # stash current state
        self._cal_img_backup = (self.img_rgb.copy(), self.img_gray.copy(),
                                self.clicks.copy())
        self._cal_clicks = []
        self.mode = "calibrate"

        # load and display the calibration image
        cal_img = Image.open(cal_path)
        cal_rgb = np.array(cal_img.convert("RGB"))
        cal_gray = np.array(cal_img.convert("L")).astype(float)
        self.img_rgb = cal_rgb
        self.img_gray = cal_gray

        self.ax_img.clear()
        self.ax_img.set_facecolor(_BG_DARK)
        self.ax_img.imshow(cal_gray, cmap="gray", aspect="equal")
        self.ax_img.set_title(
            f"CALIBRATION: {Path(cal_path).name}",
            fontsize=10, fontweight="bold", color="#ffea00",
        )
        self.ax_img.tick_params(colors=_FG_DIM, labelsize=7)
        for spine in self.ax_img.spines.values():
            spine.set_color("#ffea00")
            spine.set_linewidth(2)
        self._rubber_line = None
        self._crosshair_h = None
        self._crosshair_v = None
        self._update_status(
            "CALIBRATE: click the first end of the ruler / scale bar"
        )
        self.fig.canvas.draw_idle()

    def _finish_calibration(self):
        """Restore the original image and return to transect mode."""
        self.mode = "transect"
        self._cal_clicks = []

        if self._cal_img_backup is not None:
            self.img_rgb, self.img_gray, self.clicks = self._cal_img_backup
            self._cal_img_backup = None

        self._draw_image()
        if len(self.clicks) == 2:
            self._redraw_current_line()
            self._analyse()
        self._update_subtitle()
        self.fig.canvas.draw_idle()

    # ── save ─────────────────────────────────────────────────────────

    def _save_current(self):
        if not hasattr(self, "_last_result"):
            self._update_status("Nothing to save – draw a line first")
            return
        self._store_if_analysed()
        res = self._last_result
        stem = res["path"].stem

        # profile CSV
        csv_path = self.out_dir / f"{stem}_fft_profile.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                f"distance_{res['unit']}", "intensity",
                f"frequency_cycles_per_{res['unit']}", "power",
            ])
            n_prof = len(res["distances"])
            n_fft = len(res["freqs"])
            for i in range(max(n_prof, n_fft)):
                row = []
                if i < n_prof:
                    row += [f"{res['distances'][i]:.4f}",
                            f"{res['profile'][i]:.4f}"]
                else:
                    row += ["", ""]
                if i < n_fft:
                    row += [f"{res['freqs'][i]:.6f}",
                            f"{res['power'][i]:.4f}"]
                else:
                    row += ["", ""]
                w.writerow(row)

        # single-frame JSON output
        json_path = self.out_dir / f"{stem}_fft_result.json"
        with open(json_path, "w") as f:
            json.dump(self._result_to_dict(res), f, indent=2)

        # PNG
        png_path = self.out_dir / f"{stem}_fft_result.png"
        self.fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor=_BG_DARK)

        self._update_status(f"Saved to {self.out_dir.name}/")

    def _save_batch(self, quiet=False):
        """Export ALL accumulated results as a single JSON + summary CSV."""
        self._store_if_analysed()
        if not self.all_results:
            if not quiet:
                self._update_status("No results to batch-save")
            return

        # flatten all transects into a list
        all_transects = []
        for name, entries in self.all_results.items():
            all_transects.extend(entries)

        batch = {
            "tool": "hyphae_fft_density",
            "version": "3.0",
            "exported": datetime.now().isoformat(),
            "n_images": len(self.all_results),
            "n_transects": len(all_transects),
            "calibration_um_per_px": self.cal,
            "calibrated": self.cal_set,
            "parameters": {
                "band_width_px": self.band_width,
                "freq_cutoff": round(self.freq_cutoff, 6),
                "peak_threshold": round(self.peak_thresh, 4),
            },
            "transects": all_transects,
        }

        json_path = self.out_dir / "fft_batch_results.json"
        with open(json_path, "w") as f:
            json.dump(batch, f, indent=2)

        # summary CSV
        csv_path = self.out_dir / "fft_summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "image", "x0", "y0", "x1", "y1",
                "line_length_px", "band_width_px",
                "cal_um_per_px", "freq_cutoff", "peak_threshold",
                "peak1_freq", "peak1_wavelength", "peak1_power",
                "peak2_freq", "peak2_wavelength", "peak2_power",
                "peak3_freq", "peak3_wavelength", "peak3_power",
                "mean_intensity", "std_intensity", "unit",
            ])
            for t in all_transects:
                row = [
                    t["image"], t["x0"], t["y0"], t["x1"], t["y1"],
                    t["line_length_px"], t["band_width_px"],
                    t["calibration_um_per_px"], t["freq_cutoff"],
                    t["peak_threshold"],
                ]
                for i in range(3):
                    if i < len(t["peaks"]):
                        pk = t["peaks"][i]
                        row += [pk["frequency"], pk["wavelength"], pk["power"]]
                    else:
                        row += ["", "", ""]
                row += [t["profile_mean_intensity"],
                        t["profile_std_intensity"], t["unit"]]
                w.writerow(row)

        # full-spectrum CSV: one row per frequency bin, columns per transect
        spec_path = self.out_dir / "fft_full_spectra.csv"
        with open(spec_path, "w", newline="") as f:
            w = csv.writer(f)
            # header: frequency, then power for each transect
            labels = []
            for i, t in enumerate(all_transects):
                labels.append(f"{t['image']}_T{i+1}")
            w.writerow(["frequency"] + labels)
            # find max spectrum length
            max_bins = max(
                len(t.get("spectrum_freqs", [])) for t in all_transects
            ) if all_transects else 0
            for bi in range(max_bins):
                row = []
                # frequency from first transect that has this bin
                freq_val = ""
                for t in all_transects:
                    sf = t.get("spectrum_freqs", [])
                    if bi < len(sf):
                        freq_val = sf[bi]
                        break
                row.append(freq_val)
                for t in all_transects:
                    sp = t.get("spectrum_power", [])
                    row.append(sp[bi] if bi < len(sp) else "")
                w.writerow(row)

        if not quiet:
            self._update_status(
                f"Batch saved: {len(all_transects)} transects → {self.out_dir.name}/"
            )

    # ── session save / load ─────────────────────────────────────────

    def _save_session(self):
        """Save full session state so it can be restored later."""
        self._store_if_analysed()

        # collect per-image state: clicks + history for every image visited
        per_image = {}
        # current image
        name = self.paths[self.idx].name
        per_image[name] = {
            "clicks": [list(c) for c in self.clicks],
            "history": [
                {"p0": list(h["p0"]), "p1": list(h["p1"]),
                 "peaks": h["peaks"], "unit": h["unit"]}
                for h in self.history
            ],
        }

        session = {
            "tool": "hyphae_fft_density",
            "session_version": 1,
            "saved": datetime.now().isoformat(),
            "image_paths": [str(p) for p in self.paths],
            "current_index": self.idx,
            "calibration_um_per_px": self.cal,
            "calibrated": self.cal_set,
            "band_width_px": self.band_width,
            "freq_cutoff": round(self.freq_cutoff, 6),
            "peak_threshold": round(self.peak_thresh, 4),
            "per_image_state": per_image,
            "all_results": {
                k: v for k, v in self.all_results.items()
            },
        }

        path = self.out_dir / "fft_session.json"
        with open(path, "w") as f:
            json.dump(session, f, indent=2)
        self._update_status(f"Session saved → {path.name}")

    @classmethod
    def from_session(cls, session_path: Path, out_dir: Path):
        """Restore a saved session."""
        with open(session_path) as f:
            ses = json.load(f)

        paths = [Path(p) for p in ses["image_paths"]]
        # filter to only existing files
        paths = [p for p in paths if p.exists()]
        if not paths:
            print("No images from session still exist.")
            return None

        app = cls.__new__(cls)
        app.paths = paths
        app.out_dir = out_dir
        app.out_dir.mkdir(parents=True, exist_ok=True)
        app.idx = min(ses.get("current_index", 0), len(paths) - 1)
        app.cal = ses.get("calibration_um_per_px", 1.0)
        app.cal_set = ses.get("calibrated", False)
        app.show_gray = True
        app.band_width = ses.get("band_width_px", 5)
        app.freq_cutoff = ses.get("freq_cutoff", 0.0)
        app.peak_thresh = ses.get("peak_threshold", 0.005)
        app._auto_cutoff = False  # don't auto-set, use saved value
        app.mode = "transect"
        app._cal_clicks = []
        app._cal_img_backup = None
        app._rubber_line = None
        app._crosshair_h = None
        app._crosshair_v = None
        app._help_visible = False
        app._help_artists = []
        app._press_xy = None
        app.clicks = []
        app.history = []
        app.all_results = ses.get("all_results", {})

        # restore per-image state
        app._saved_per_image = ses.get("per_image_state", {})

        # build the figure (same as __init__)
        import matplotlib.gridspec as gridspec
        app.fig = plt.figure(figsize=(16, 9.5), facecolor=_BG_DARK)
        gs = gridspec.GridSpec(
            3, 2, figure=app.fig,
            width_ratios=[1.55, 1],
            height_ratios=[1.0, 0.35, 1.0],
            hspace=0.38, wspace=0.28,
            left=0.05, right=0.97, top=0.92, bottom=0.08,
        )
        app.ax_img = app.fig.add_subplot(gs[:, 0])
        app.ax_prof = app.fig.add_subplot(gs[0, 1])
        app.ax_strip = app.fig.add_subplot(gs[1, 1])
        app.ax_fft = app.fig.add_subplot(gs[2, 1])

        app.fig.text(
            0.50, 0.97, "Hyphal Density – FFT Analyser",
            fontsize=13, fontweight="bold", color=_FG_TEXT,
            ha="center", va="top",
        )
        app.subtitle_text = app.fig.text(
            0.50, 0.945, "", fontsize=9, color=_FG_DIM,
            ha="center", va="top",
        )

        # buttons (same as __init__)
        btn_y, btn_h = 0.015, 0.035
        btn_specs = [
            (0.030, 0.040, "◀ Prev"),
            (0.075, 0.040, "Next ▶"),
            (0.120, 0.045, "Reset"),
            (0.170, 0.040, "Save"),
            (0.215, 0.050, "SaveAll"),
            (0.270, 0.055, "Calibrate"),
            (0.330, 0.040, "Bnd-"),
            (0.375, 0.040, "Bnd+"),
            (0.420, 0.050, "Cut-"),
            (0.475, 0.050, "Cut+"),
            (0.530, 0.040, "Thr-"),
            (0.575, 0.040, "Thr+"),
            (0.620, 0.040, "Help"),
        ]
        app.btns = {}
        for bx, bw, label in btn_specs:
            ax_btn = app.fig.add_axes([bx, btn_y, bw, btn_h])
            ax_btn.set_facecolor("#333333")
            btn = Button(ax_btn, label, color="#333333", hovercolor="#555555")
            btn.label.set_color(_FG_TEXT)
            btn.label.set_fontsize(7)
            app.btns[label] = btn

        app.btns["◀ Prev"].on_clicked(lambda _: app._go_prev())
        app.btns["Next ▶"].on_clicked(lambda _: app._go_next())
        app.btns["Reset"].on_clicked(lambda _: app._reset())
        app.btns["Save"].on_clicked(lambda _: app._save_current())
        app.btns["SaveAll"].on_clicked(lambda _: app._save_batch())
        app.btns["Calibrate"].on_clicked(lambda _: app._ask_calibration())
        app.btns["Bnd-"].on_clicked(lambda _: app._change_band(-2))
        app.btns["Bnd+"].on_clicked(lambda _: app._change_band(+2))
        app.btns["Cut-"].on_clicked(lambda _: app._change_cutoff(-1))
        app.btns["Cut+"].on_clicked(lambda _: app._change_cutoff(+1))
        app.btns["Thr-"].on_clicked(lambda _: app._change_thresh(-1))
        app.btns["Thr+"].on_clicked(lambda _: app._change_thresh(+1))
        app.btns["Help"].on_clicked(lambda _: app._toggle_help())

        app.status_text = app.fig.text(
            0.74, 0.022, "", fontsize=8, color=_FG_DIM,
            ha="left", va="bottom", family="monospace",
        )
        app.param_text = app.fig.text(
            0.97, 0.022, "", fontsize=7.5, color="#80cbc4",
            ha="right", va="bottom", family="monospace",
        )

        app.fig.canvas.mpl_connect("button_press_event", app._on_press)
        app.fig.canvas.mpl_connect("button_release_event", app._on_release)
        app.fig.canvas.mpl_connect("key_press_event", app._on_key)
        app.fig.canvas.mpl_connect("motion_notify_event", app._on_motion)

        # load current image and restore its state
        app._load_image()
        app._restore_image_state()
        app._update_param_label()
        app._update_status(f"Session restored: {len(app.all_results)} images")
        plt.show()
        return app

    def _restore_image_state(self):
        """Restore clicks/history for the current image from saved state."""
        name = self.paths[self.idx].name
        state = getattr(self, "_saved_per_image", {}).get(name)
        if not state:
            return
        self.history = [
            {"p0": tuple(h["p0"]), "p1": tuple(h["p1"]),
             "peaks": h["peaks"], "unit": h["unit"]}
            for h in state.get("history", [])
        ]
        clicks_raw = state.get("clicks", [])
        self.clicks = [tuple(c) for c in clicks_raw]
        self._draw_image()
        if len(self.clicks) == 2:
            self._redraw_current_line()
            self._analyse()

    # ── help overlay ─────────────────────────────────────────────────

    def _toggle_help(self):
        if self._help_visible:
            for a in self._help_artists:
                a.remove()
            self._help_artists.clear()
            self._help_visible = False
            self.fig.canvas.draw_idle()
            return
        help_lines = [
            "─── CONTROLS ───────────────────────────────",
            "",
            "  Click – Click    Draw transect line (auto-chains)",
            "  r                Reset current transect",
            "  n                Store transect & draw next",
            "  s                Save current frame (CSV+JSON+PNG)",
            "  S                Batch-save ALL results",
            "  c                Calibrate: pick ruler image,",
            "                   click two ends, enter mm",
            "",
            "  w / W            Band width  −/+  (noise avg)",
            "  f / F            Low-freq cutoff  −/+",
            "  t / T            Peak threshold   −/+",
            "",
            "  g                Toggle grayscale / colour",
            "  ← / →            Previous / Next image",
            "  h                Toggle this help",
            "  q / Esc          Quit",
            "",
            "─── SENSITIVITY ────────────────────────────",
            "",
            "  Low-freq cutoff (f/F):",
            "    Filters out slow intensity gradients.",
            "    Increase until the big low-freq peak",
            "    is excluded and fine peaks emerge.",
            "",
            "  Peak threshold (t/T):",
            "    Fraction of max power required.",
            "    Lower = more peaks detected.",
            "    Raise to keep only the strongest.",
            "",
            "─── OUTPUT (in OSF/FFT/) ───────────────────",
            "",
            "  fft_batch_results.json  ← batch analysis input",
            "  fft_summary.csv         ← spreadsheet-ready",
            "  {name}_fft_profile.csv  ← per-frame data",
            "  {name}_fft_result.json  ← per-frame JSON",
            "  {name}_fft_result.png   ← figure snapshot",
        ]
        text = "\n".join(help_lines)
        bg = self.fig.add_axes(
            [0.15, 0.10, 0.70, 0.80], facecolor=_BG_DARK, zorder=100,
        )
        bg.set_alpha(0.94)
        bg.set_xticks([])
        bg.set_yticks([])
        for spine in bg.spines.values():
            spine.set_color("#80cbc4")
            spine.set_linewidth(1.5)
        t = bg.text(
            0.04, 0.97, text,
            transform=bg.transAxes, fontsize=9.5, fontfamily="monospace",
            color=_FG_TEXT, va="top", ha="left", linespacing=1.3,
        )
        self._help_artists = [bg, t]
        self._help_visible = True
        self.fig.canvas.draw_idle()

    # ── status helpers ───────────────────────────────────────────────

    def _update_subtitle(self):
        n = len(self.paths)
        nav = f"  [{self.idx + 1}/{n}]" if n > 1 else ""
        cal = f"  •  {self.cal:.3f} µm/px" if self.cal_set else "  •  uncalibrated"
        self.subtitle_text.set_text(f"{self.paths[self.idx].name}{nav}{cal}")
        self.fig.canvas.draw_idle()

    def _update_status(self, msg=""):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _update_param_label(self):
        self.param_text.set_text(
            f"band:{self.band_width}px  "
            f"cutoff:{self.freq_cutoff:.4f}  "
            f"thr:{self.peak_thresh:.0%}"
        )
        self.fig.canvas.draw_idle()


# ── entry point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FFT-based hyphal density analysis along drawn lines."
    )
    parser.add_argument(
        "paths", nargs="*", default=[],
        help="Image files/folders (opens multi-file dialog if omitted).",
    )
    parser.add_argument(
        "-o", "--output", default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to fft_session.json to restore a previous session.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).expanduser().resolve()

    # resume mode
    if args.resume:
        session_path = Path(args.resume).expanduser().resolve()
        if not session_path.exists():
            sys.exit(f"Session file not found: {session_path}")
        print(f"Resuming session from {session_path.name}")
        HyphaeFFTApp.from_session(session_path, out_dir)
        return

    if args.paths:
        images = []
        for p in args.paths:
            pp = Path(p).expanduser().resolve()
            images.extend(_collect_images_from_path(pp))
    else:
        chosen = _macos_multi_file_dialog()
        if not chosen:
            folder = _macos_folder_dialog()
            if folder:
                chosen = [folder]
        if not chosen:
            sys.exit("No files selected.")
        images = []
        for c in chosen:
            images.extend(_collect_images_from_path(Path(c)))

    if not images:
        sys.exit("No images found.")

    # deduplicate, preserve order
    seen = set()
    unique = []
    for im in images:
        if im not in seen:
            seen.add(im)
            unique.append(im)

    print(f"Loaded {len(unique)} image(s). Output → {out_dir}")
    HyphaeFFTApp(unique, out_dir)


if __name__ == "__main__":
    main()
