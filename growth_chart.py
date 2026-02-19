#!/usr/bin/env python3
"""
Publication-quality growth chart with adult height prediction.

Generates a Nature/Lancet-quality figure showing:
  - CDC girls growth reference percentile bands (0-18 years)
  - Sena's measured data points
  - Projected growth trajectory (Cole & Wright 2011)
  - Expanding/contracting uncertainty cone
  - Predicted adult height marker

Output: sena_growth_chart.png (300 DPI) and sena_growth_chart.pdf
"""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from predictor import get_correlation, zscore_to_height, height_to_zscore, ADULT_MEAN, ADULT_SD

BASE_DIR = Path(__file__).parent

# ── Sena's data ──────────────────────────────────────────────────────
MEASURED_POINT = (1.010, 78.0)  # (age_years, height_cm) - exact 12-month

# Approximate earlier visits (read from WHO chart in visit summary PDF)
# Sena rose from ~72nd percentile at birth to 93rd by 12 months
APPROX_VISITS = [
    (0.0,    50.8),   # birth  (~20.0", ~72nd CDC pct)
    (2/12,   59.1),   # 2 months (~23.3", ~84th CDC pct)
    (4/12,   64.6),   # 4 months (~25.4", ~89th CDC pct)
    (6/12,   68.5),   # 6 months (~27.0", ~90th CDC pct)
    (8/12,   72.1),   # 8 months (~28.4", ~92nd CDC pct)
]

SEX = "F"
Z_CHILD = height_to_zscore(MEASURED_POINT[1], MEASURED_POINT[0], SEX)
R_NOW = get_correlation(MEASURED_POINT[0], SEX)
Z_ADULT = R_NOW * Z_CHILD

# ── Color palette ────────────────────────────────────────────────────
C = {
    # Percentile bands (nested, blue-gray)
    "band_outer": "#8BAAC4",
    "band_mid":   "#A3BDD1",
    "band_inner": "#BBCFDE",
    "band_core":  "#D2E1EC",
    # Percentile lines
    "pct_50":     "#2C5F8A",
    "pct_major":  "#5B8BAE",
    "pct_outer":  "#85A8C2",
    # Sena
    "sena_exact": "#C0392B",
    "sena_approx":"#E07A5F",
    "trajectory": "#C0392B",
    # Prediction cone
    "cone":       "#E8963E",
    # Adult marker
    "adult":      "#1A8A7A",
    # Structure
    "grid":       "#EBEBEB",
    "spine":      "#BBBBBB",
    "text":       "#2D2D2D",
    "subtext":    "#777777",
}

# ── 1. Load CDC percentile curves for girls ──────────────────────────

def _read_csv(filename, sex_code=2):
    """Read CDC CSV, return list of dicts for one sex."""
    rows = []
    with open(BASE_DIR / filename) as f:
        for row in csv.DictReader(f):
            if int(row["Sex"]) == sex_code:
                rows.append({k: float(row[k]) for k in
                    ["Agemos", "L", "M", "S", "P3", "P5", "P10", "P25",
                     "P50", "P75", "P90", "P95", "P97"]})
    return rows


def load_cdc_percentiles():
    """Load CDC female percentile curves, blending infant/stature at 22-26 months.

    Returns: (ages_years, dict of percentile_name -> height_cm_array)
    """
    infant = _read_csv("lenageinf.csv")
    stature = _read_csv("statage.csv")

    pct_names = ["P3", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P97"]

    # Build interpolators for each dataset and percentile
    inf_ages = [r["Agemos"] for r in infant]
    sta_ages = [r["Agemos"] for r in stature]

    inf_interp = {p: interp1d(inf_ages, [r[p] for r in infant], kind="linear",
                              fill_value="extrapolate") for p in pct_names}
    sta_interp = {p: interp1d(sta_ages, [r[p] for r in stature], kind="linear",
                              fill_value="extrapolate") for p in pct_names}

    # Generate fine age grid: 0 to 216 months (18 years)
    ages_mo = np.concatenate([
        np.linspace(0, 36, 360),       # 0-3 years: fine resolution
        np.linspace(36.1, 216, 500),   # 3-18 years
    ])

    result = {}
    for p in pct_names:
        vals = np.empty_like(ages_mo)
        for i, a in enumerate(ages_mo):
            if a < 22:
                vals[i] = inf_interp[p](a)
            elif a > 26:
                vals[i] = sta_interp[p](a)
            else:
                # Blend
                w = (a - 22) / 4.0
                vals[i] = (1 - w) * inf_interp[p](a) + w * sta_interp[p](a)
        result[p] = vals

    ages_yr = ages_mo / 12.0
    return ages_yr, result


# ── 2. Compute trajectory and uncertainty cone ───────────────────────

def compute_trajectory():
    """Compute projected center line and confidence bands from age 1 to 18.

    Trajectory model:
      The z-score smoothly transitions from z_child (measured) to z_adult
      (predicted) using cumulative growth fraction as the interpolation weight.
      This ensures the trajectory NEVER crosses percentile lines upward —
      it monotonically regresses toward the mean.

    Uncertainty model:
      SE accumulates proportionally to sqrt of cumulative growth completed.
      This naturally produces wider uncertainty during the pubertal growth
      spurt (more growth = more accumulated uncertainty).

    Returns: (ages, center, lower_80, upper_80, lower_95, upper_95) all in cm
    """
    ages = np.linspace(MEASURED_POINT[0], 18.0, 600)

    # Total adult prediction SE in z-score units
    se_z_adult = math.sqrt(max(0, 1 - R_NOW ** 2))

    # Cumulative growth fractions: how much of the remaining growth
    # (from measurement to adulthood) has occurred at each age
    p50_now = zscore_to_height(0, MEASURED_POINT[0], SEX)
    p50_end = zscore_to_height(0, 18.0, SEX)
    total_remaining_growth = p50_end - p50_now

    center = np.empty_like(ages)
    lo80, hi80 = np.empty_like(ages), np.empty_like(ages)
    lo95, hi95 = np.empty_like(ages), np.empty_like(ages)

    for i, t in enumerate(ages):
        # Fraction of remaining growth completed at age t
        p50_t = zscore_to_height(0, t, SEX)
        frac = (p50_t - p50_now) / total_remaining_growth
        frac = min(1.0, max(0.0, frac))

        # Center z-score: smooth monotonic interpolation from z_child to z_adult
        # Uses growth fraction so regression accelerates during growth spurt
        z_c = Z_CHILD + (Z_ADULT - Z_CHILD) * frac

        # SE grows with sqrt of cumulative growth fraction
        se_z = se_z_adult * math.sqrt(frac)

        center[i] = zscore_to_height(z_c, t, SEX)
        lo80[i] = zscore_to_height(z_c - 1.28 * se_z, t, SEX)
        hi80[i] = zscore_to_height(z_c + 1.28 * se_z, t, SEX)
        lo95[i] = zscore_to_height(z_c - 1.96 * se_z, t, SEX)
        hi95[i] = zscore_to_height(z_c + 1.96 * se_z, t, SEX)

    return ages, center, lo80, hi80, lo95, hi95


# ── 3. Helper: cm → feet/inches label ────────────────────────────────

def cm_to_label(cm):
    total_in = cm / 2.54
    ft = int(total_in // 12)
    inch = total_in % 12
    inch_r = int(round(inch))
    if inch_r == 12:
        ft += 1
        inch_r = 0
    return f"{ft}'{inch_r}\""


# ── 4. Build the figure ──────────────────────────────────────────────

def create_figure():
    # -- Setup --
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "mathtext.default": "regular",
    })

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # -- Load data --
    ages_yr, pcts = load_cdc_percentiles()
    traj_ages, traj_center, lo80, hi80, lo95, hi95 = compute_trajectory()

    # -- Axis limits --
    ax.set_xlim(-0.2, 19.2)
    ax.set_ylim(42, 185)  # cm range (~16.5" to ~72.8")

    # -- Percentile bands (outermost first) --
    band_specs = [
        ("P3",  "P97", C["band_outer"], 0.20),
        ("P5",  "P95", C["band_mid"],   0.22),
        ("P10", "P90", C["band_inner"], 0.25),
        ("P25", "P75", C["band_core"],  0.30),
    ]
    for lo_p, hi_p, color, alpha in band_specs:
        ax.fill_between(ages_yr, pcts[lo_p], pcts[hi_p],
                        color=color, alpha=alpha, linewidth=0, zorder=1)

    # -- Percentile lines --
    line_specs = [
        ("P97", 0.6, C["pct_outer"], 0.50, "97th"),
        ("P90", 0.7, C["pct_major"], 0.55, "90th"),
        ("P75", 0.7, C["pct_major"], 0.55, "75th"),
        ("P50", 1.2, C["pct_50"],    0.70, "50th"),
        ("P25", 0.7, C["pct_major"], 0.55, "25th"),
        ("P10", 0.7, C["pct_major"], 0.55, "10th"),
        ("P3",  0.6, C["pct_outer"], 0.50, "3rd"),
    ]
    for pname, lw, color, alpha, _ in line_specs:
        ax.plot(ages_yr, pcts[pname], color=color, linewidth=lw,
                alpha=alpha, zorder=2)

    # -- Percentile labels at right edge --
    for pname, _, color, _, label in line_specs:
        y_val = pcts[pname][-1]
        fw = "bold" if pname == "P50" else "medium"
        fs = 7.5 if pname == "P50" else 6.5
        ax.text(18.35, y_val, label, fontsize=fs, color=color,
                va="center", ha="left", fontweight=fw, zorder=2)

    # -- Uncertainty cone --
    ax.fill_between(traj_ages, lo95, hi95,
                    color=C["cone"], alpha=0.15, linewidth=0, zorder=3,
                    label="95% prediction interval")
    ax.fill_between(traj_ages, lo80, hi80,
                    color=C["cone"], alpha=0.28, linewidth=0, zorder=3,
                    label="80% prediction interval")
    # Thin boundary lines for crisp cone edges
    ax.plot(traj_ages, lo95, color=C["cone"], linewidth=0.5, alpha=0.30, zorder=3)
    ax.plot(traj_ages, hi95, color=C["cone"], linewidth=0.5, alpha=0.30, zorder=3)
    ax.plot(traj_ages, lo80, color=C["cone"], linewidth=0.4, alpha=0.20, zorder=3)
    ax.plot(traj_ages, hi80, color=C["cone"], linewidth=0.4, alpha=0.20, zorder=3)

    # -- Trajectory center line (with subtle glow) --
    ax.plot(traj_ages, traj_center,
            color=C["trajectory"], linewidth=3.5, alpha=0.12,
            zorder=4, solid_capstyle="round")
    ax.plot(traj_ages, traj_center,
            color=C["trajectory"], linewidth=2.0, alpha=0.85,
            zorder=4, label="Projected trajectory")

    # -- Connect approximate points to measured point --
    all_ages = [a for a, _ in APPROX_VISITS] + [MEASURED_POINT[0]]
    all_heights = [h for _, h in APPROX_VISITS] + [MEASURED_POINT[1]]
    ax.plot(all_ages, all_heights,
            color=C["sena_approx"], linewidth=1.0, alpha=0.40, zorder=4,
            linestyle="-")

    # -- Approximate data points --
    approx_a = [a for a, _ in APPROX_VISITS]
    approx_h = [h for _, h in APPROX_VISITS]
    ax.scatter(approx_a, approx_h,
               s=36, facecolors="none", edgecolors=C["sena_approx"],
               linewidths=1.5, zorder=5, label="Earlier visits (est.)")

    # -- Exact 12-month measurement --
    ax.scatter([MEASURED_POINT[0]], [MEASURED_POINT[1]],
               s=80, color=C["sena_exact"], edgecolors="white",
               linewidths=1.5, zorder=6, label="Measured (12 mo)")

    # -- Name label near data --
    ax.annotate(
        "Sena",
        xy=(MEASURED_POINT[0], MEASURED_POINT[1]),
        xytext=(2.0, MEASURED_POINT[1] + 4),
        fontsize=10, fontweight="bold", color=C["sena_exact"],
        arrowprops=dict(arrowstyle="-", color=C["sena_exact"],
                        lw=0.8, alpha=0.5),
        zorder=6,
    )

    # -- Adult prediction marker at age 18 --
    adult_h = traj_center[-1]  # height at age 18
    ax.scatter([18], [adult_h], s=120, color=C["adult"],
               edgecolors="white", linewidths=2.0, zorder=7,
               marker="D")

    # -- 95% CI bar at age 18 --
    ci_lo_18 = lo95[-1]
    ci_hi_18 = hi95[-1]
    ax.plot([18.15, 18.15], [ci_lo_18, ci_hi_18],
            color=C["adult"], linewidth=2.0, alpha=0.6, zorder=7)
    ax.plot([17.95, 18.35], [ci_lo_18, ci_lo_18],
            color=C["adult"], linewidth=1.5, alpha=0.6, zorder=7)
    ax.plot([17.95, 18.35], [ci_hi_18, ci_hi_18],
            color=C["adult"], linewidth=1.5, alpha=0.6, zorder=7)

    # Adult annotation
    adult_ft = cm_to_label(adult_h)
    lo_ft = cm_to_label(ci_lo_18)
    hi_ft = cm_to_label(ci_hi_18)
    ax.annotate(
        f"Predicted: {adult_ft}\n95% CI: {lo_ft} – {hi_ft}",
        xy=(18, adult_h),
        xytext=(12.8, adult_h + 12),
        fontsize=9.5, fontweight="bold", color=C["adult"],
        arrowprops=dict(arrowstyle="->", color=C["adult"],
                        lw=1.2, connectionstyle="arc3,rad=-0.15"),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=C["adult"], alpha=0.97, linewidth=0.8),
        zorder=8,
    )

    # -- Y-axis: feet and inches --
    tick_inches = list(range(18, 74, 6))  # 1'6" through 6'0"
    tick_cm = [i * 2.54 for i in tick_inches]
    tick_labels = [f"{i // 12}'{i % 12}\"" for i in tick_inches]
    ax.set_yticks(tick_cm)
    ax.set_yticklabels(tick_labels)

    minor_inches = list(range(18, 74, 2))
    minor_cm = [i * 2.54 for i in minor_inches]
    ax.set_yticks(minor_cm, minor=True)

    ax.set_ylabel("Height", fontsize=13, fontweight="medium",
                  color=C["text"])

    # -- X-axis --
    ax.set_xticks(range(0, 19, 1))
    ax.set_xticks(np.arange(0, 18.5, 0.5), minor=True)
    ax.set_xlabel("Age (years)", fontsize=13, fontweight="medium",
                  color=C["text"])

    # -- Grid --
    ax.grid(True, which="major", axis="both", color=C["grid"],
            linewidth=0.5, alpha=0.9, zorder=0)
    ax.grid(True, which="minor", axis="both", color=C["grid"],
            linewidth=0.3, alpha=0.4, zorder=0)

    # -- Spines --
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C["spine"])
    ax.spines["bottom"].set_color(C["spine"])
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)

    ax.tick_params(axis="both", which="both", colors=C["text"],
                   direction="out", length=4, width=0.6)
    ax.tick_params(axis="both", which="minor", length=2, width=0.4)

    # -- Title --
    ax.set_title("Height Growth Trajectory with Adult Height Prediction",
                 fontsize=15, fontweight="bold", color=C["text"],
                 pad=20, loc="left")
    ax.text(0, 1.02,
            "CDC Girls Growth Reference  •  Projection: Cole & Wright, "
            "Ann Hum Biol 2011;38:662–8",
            transform=ax.transAxes, fontsize=8.5, color=C["subtext"],
            style="italic", va="bottom")

    # -- Legend --
    legend = ax.legend(
        loc="upper left", frameon=True, framealpha=0.95,
        edgecolor=C["spine"], fancybox=False,
        borderpad=0.8, labelspacing=0.6,
        handlelength=1.8, handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.5)
    for text in legend.get_texts():
        text.set_color(C["text"])

    # -- Bottom caption --
    fig.text(
        0.10, 0.01,
        "Shaded bands: CDC percentiles (P3–P97).  "
        "Amber cone: prediction interval (inner 80%, outer 95%).  "
        "Uncertainty grows with cumulative growth remaining (Cole & Wright 2011).  "
        "Approximate early measurements estimated from visit growth chart.",
        fontsize=7.5, color="#999999", style="italic",
    )

    # -- Layout --
    fig.tight_layout(rect=[0, 0.035, 1, 1])

    # -- Save --
    out_png = BASE_DIR / "sena_growth_chart.png"
    out_pdf = BASE_DIR / "sena_growth_chart.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    fig.savefig(out_pdf, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    create_figure()
