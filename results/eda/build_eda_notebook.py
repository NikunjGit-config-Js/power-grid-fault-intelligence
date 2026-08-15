"""Build the EDA notebook for Phase 1: Complete Data Analysis.

This script creates results/eda/eda_report.ipynb with all 7 subsections
defined in docs/PROJECT_SPEC.md lines 215-425.
"""

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
})

cells = []

# ---------------------------------------------------------------------------
# Helper to add markdown and code cells
# ---------------------------------------------------------------------------
def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

# ===========================================================================
# Title
# ===========================================================================
md("""# Phase 1 - Complete Data Analysis

**Dataset:** `classData.csv` - Power-grid fault detection/classification

**Objective:** Systematic EDA following `docs/PROJECT_SPEC.md` sections 1.1-1.7.

**Rules:**
- All calculations are descriptive/read-only - no rows deleted, no columns removed.
- Outliers are identified and analysed, not deleted unless clear measurement corruption.
- Near-zero-variance columns are identified only, not removed.
- No arbitrary engineering thresholds for physical bounds.
- All figures saved reproducibly to `results/eda/figures/`.
""")

# ===========================================================================
# Imports
# ===========================================================================
code("""import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# Project root
ROOT = Path.cwd()
if not (ROOT / "classData.csv").exists():
    ROOT = Path(r"C:\\Users\\ASUS\\Downloads\\power-grid-fault-intelligence")

DATA_PATH = ROOT / "classData.csv"
FIG_DIR = ROOT / "results" / "eda" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"ROOT:      {ROOT}")
print(f"DATA_PATH: {DATA_PATH}")
print(f"FIG_DIR:   {FIG_DIR}")
""")

# ===========================================================================
# 1.1 Dataset Overview
# ===========================================================================
md("""---
## 1.1 Dataset Overview
""")

code("""df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\\nColumns: {list(df.columns)}")
print(f"\\nDtypes:")
print(df.dtypes)
print(f"\\nFirst 5 rows:")
df.head()
""")

code("""print(f"Missing values per column:")
print(df.isnull().sum())
print(f"\\nTotal missing: {df.isnull().sum().sum()}")
print(f"\\nDuplicate rows: {df.duplicated().sum()}")
""")

code("""# Physical meaning of columns
physical_meaning = {
    "Ia": "Phase A current (Amperes) - instantaneous current waveform on phase A",
    "Ib": "Phase B current (Amperes) - instantaneous current waveform on phase B",
    "Ic": "Phase C current (Amperes) - instantaneous current waveform on phase C",
    "Va": "Phase A voltage (kV or p.u.) - instantaneous voltage waveform on phase A",
    "Vb": "Phase B voltage (kV or p.u.) - instantaneous voltage waveform on phase B",
    "Vc": "Phase C voltage (kV or p.u.) - instantaneous voltage waveform on phase C",
    "G": "Ground fault indicator bit (1 = ground fault involved)",
    "C": "Phase C fault indicator bit (1 = phase C involved in fault)",
    "B": "Phase B fault indicator bit (1 = phase B involved in fault)",
    "A": "Phase A fault indicator bit (1 = phase A involved in fault)",
}

for col, desc in physical_meaning.items():
    print(f"  {col:>3s}: {desc}")
""")

code("""# Observed fault codes and class distribution
fault_codes = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
code_counts = fault_codes.value_counts().sort_index()

print("Observed fault codes (G,C,B,A -> code -> count -> percentage):")
print("-" * 65)
for code_val, count in code_counts.items():
    pct = 100 * count / len(df)
    print(f"  {code_val}  ->  {count:>5d} samples  ({pct:5.1f}%)")

print(f"\\nNumber of classes: {len(code_counts)}")
print(f"Total samples: {len(df)}")
""")

# ===========================================================================
# 1.2 Data Quality Analysis
# ===========================================================================
md("""---
## 1.2 Data Quality Analysis
""")

code("""# Infinite values
measurement_cols = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
inf_counts = np.isinf(df[measurement_cols]).sum()
print("Infinite values per measurement column:")
print(inf_counts)
print(f"\\nTotal infinite values: {inf_counts.sum()}")
""")

code("""# Suspicious / extreme values - descriptive only
# No arbitrary thresholds. Report what exists.
print("Descriptive statistics for measurement columns:")
print("=" * 80)
desc = df[measurement_cols].describe().T
desc["skewness"] = df[measurement_cols].skew()
desc["kurtosis"] = df[measurement_cols].kurtosis()
desc["IQR"] = desc["75%"] - desc["25%"]
desc["n_below_q1_1.5"] = (df[measurement_cols] < desc["25%"] - 1.5 * desc["IQR"]).sum().values
desc["n_above_q3_1.5"] = (df[measurement_cols] > desc["75%"] + 1.5 * desc["IQR"]).sum().values
print(desc.to_string())
""")

code("""# Near-zero variance - identify only, do not remove
variances = df[measurement_cols].var()
print("Variance per measurement column:")
print(variances)
print("\\nNZV columns (variance < 1e-6):", list(variances[variances < 1e-6].index) if (variances < 1e-6).any() else "None")
""")

code("""# Constant columns - identify only
nunique = df.nunique()
constant_cols = nunique[nunique <= 1].index.tolist()
print("Constant columns:", constant_cols if constant_cols else "None")
""")

code("""# Suspiciously duplicated signals - high pairwise correlation
corr_raw = df[measurement_cols].corr()
print("Pairwise correlation matrix (raw measurements):")
print(corr_raw.round(3))
print("\\nPairs with |r| > 0.99 (potentially redundant):")
found = False
for i in range(len(measurement_cols)):
    for j in range(i + 1, len(measurement_cols)):
        r = corr_raw.iloc[i, j]
        if abs(r) > 0.99:
            print(f"  {measurement_cols[i]} <-> {measurement_cols[j]}: r = {r:.4f}")
            found = True
if not found:
    print("  None")
""")

code("""# Invalid labels - check for unknown G,C,B/A codes
known_codes = {"0000", "1001", "0110", "0111", "1011", "1111"}
all_codes = set(fault_codes.unique())
unknown_codes = all_codes - known_codes
print(f"Known codes:  {sorted(known_codes)}")
print(f"Found codes:  {sorted(all_codes)}")
print(f"Unknown codes: {sorted(unknown_codes) if unknown_codes else 'None'}")
""")

# ===========================================================================
# 1.3 Class Distribution
# ===========================================================================
md("""---
## 1.3 Class Distribution
""")

code("""# Map fault codes to human-readable labels
FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "LL-G BC",
    "0111": "Three-Phase",
    "1011": "LG A+B-G",
    "1111": "3P-G",
}

df["fault_code"] = fault_codes
df["fault_label"] = df["fault_code"].map(FAULT_CODE_TO_LABEL)

class_dist = df["fault_label"].value_counts().sort_values(ascending=False)
print("Class distribution:")
print(class_dist)
print(f"\\nPercentages:")
print((100 * class_dist / len(df)).round(2))
""")

code("""# Bar plot: Fault Class vs Sample Count
fig, ax = plt.subplots(figsize=(10, 5))
colors = sns.color_palette("Set2", len(class_dist))
bars = ax.bar(class_dist.index, class_dist.values, color=colors, edgecolor="black", linewidth=0.5)

for bar, count in zip(bars, class_dist.values):
    pct = 100 * count / len(df)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
            f"{count}\\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("Fault Class", fontsize=12)
ax.set_ylabel("Sample Count", fontsize=12)
ax.set_title("Fault Class Distribution", fontsize=14)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(FIG_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'class_distribution.png'}")
""")

code("""# Binary summary: Fault vs No Fault
no_fault = (df["fault_code"] == "0000").sum()
fault = len(df) - no_fault
binary_counts = pd.Series({"No Fault": no_fault, "Fault": fault})

fig, ax = plt.subplots(figsize=(5, 4))
binary_counts.plot(kind="bar", color=["#2ecc71", "#e74c3c"], edgecolor="black", ax=ax)
ax.set_ylabel("Sample Count")
ax.set_title("Binary: Fault vs No Fault")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, (label, val) in enumerate(binary_counts.items()):
    ax.text(i, val + 30, f"{val} ({100*val/len(df):.1f}%)", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIG_DIR / "binary_fault_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"No Fault: {no_fault} ({100*no_fault/len(df):.1f}%)")
print(f"Fault:    {fault} ({100*fault/len(df):.1f}%)")
""")

# ===========================================================================
# 1.4 Univariate Analysis
# ===========================================================================
md("""---
## 1.4 Univariate Analysis

For every raw electrical measurement (Ia, Ib, Ic, Va, Vb, Vc):
- Histogram + KDE
- Box plot
- Descriptive statistics
""")

code("""# Descriptive statistics table for all 6 measurements
stats_df = df[measurement_cols].describe().T
stats_df["skewness"] = df[measurement_cols].skew()
stats_df["kurtosis"] = df[measurement_cols].kurtosis()
stats_df["median"] = df[measurement_cols].median()
stats_df["IQR"] = stats_df["75%"] - stats_df["25%"]
stats_df = stats_df[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max", "IQR", "skewness", "kurtosis"]]
print(stats_df.round(4).to_string())
""")

code("""# Histogram + KDE + Box plot for each measurement
for col in measurement_cols:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram + KDE
    axes[0].hist(df[col], bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="black", linewidth=0.5)
    df[col].plot.kde(ax=axes[0], color="darkred", linewidth=2)
    axes[0].set_title(f"{col} - Histogram + KDE")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")

    # Box plot
    axes[1].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="black"),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_title(f"{col} - Box Plot")
    axes[1].set_ylabel(col)

    # Box plot by fault class
    data_by_class = [df.loc[df["fault_label"] == lbl, col].values for lbl in sorted(df["fault_label"].unique())]
    bp = axes[2].boxplot(data_by_class, vert=True, patch_artist=True,
                         labels=[l[:12] for l in sorted(df["fault_label"].unique())])
    class_colors = sns.color_palette("Set2", len(df["fault_label"].unique()))
    for patch, color in zip(bp["boxes"], class_colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)
    axes[2].set_title(f"{col} - Box Plot by Fault Class")
    axes[2].set_ylabel(col)
    axes[2].tick_params(axis="x", rotation=45)

    plt.suptitle(f"Univariate Analysis: {col}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"univariate_{col}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {FIG_DIR / f'univariate_{col}.png'}")
""")

# ===========================================================================
# 1.5 Outlier Analysis
# ===========================================================================
md("""---
## 1.5 Outlier Analysis

Outliers are identified and classified conceptually. **No rows are deleted.**
Values are classified as:
- **Potential measurement corruption** - only if clear evidence exists
- **Physically meaningful fault extremes** - high current during fault = signal
""")

code("""# IQR-based outlier identification per measurement
print("IQR-based outlier counts (identified only, not removed):")
print("=" * 70)
outlier_summary = {}
for col in measurement_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_below = (df[col] < lower).sum()
    n_above = (df[col] > upper).sum()
    n_total = n_below + n_above
    outlier_summary[col] = {
        "Q1": Q1, "Q3": Q3, "IQR": IQR,
        "lower_bound": lower, "upper_bound": upper,
        "n_below": n_below, "n_above": n_above, "n_total": n_total,
        "pct": 100 * n_total / len(df)
    }
    print(f"  {col}: Q1={Q1:.4f}, Q3={Q3:.4f}, IQR={IQR:.4f}")
    print(f"       bounds: [{lower:.4f}, {upper:.4f}]")
    print(f"       outliers: {n_below} below + {n_above} above = {n_total} ({100*n_total/len(df):.1f}%)")
    print()
""")

code("""# Classification: fault extremes vs potential corruption
# Check if outlier values overlap with high-current fault samples
print("Outlier overlap with fault classes:")
print("=" * 70)
for col in ["Ia", "Ib", "Ic"]:
    info = outlier_summary[col]
    if info["n_total"] > 0:
        outlier_mask = (df[col] < info["lower_bound"]) | (df[col] > info["upper_bound"])
        fault_in_outliers = df.loc[outlier_mask, "fault_code"].value_counts()
        print(f"\\n{col} outliers by fault code:")
        for code_val, count in fault_in_outliers.items():
            label = FAULT_CODE_TO_LABEL.get(code_val, "Unknown")
            print(f"  {code_val} ({label}): {count}")
        # Check if all outliers belong to fault classes
        no_fault_outliers = (df.loc[outlier_mask, "fault_code"] == "0000").sum()
        print(f"  -> No-fault samples in outliers: {no_fault_outliers}")
        if no_fault_outliers == 0:
            print(f"  -> Classification: PHYSICALLY MEANINGFUL FAULT EXTREMES")
        else:
            print(f"  -> Classification: MIXED - some non-fault outliers exist")
""")

code("""# Box plots per fault class for current measurements
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(["Ia", "Ib", "Ic"]):
    fault_labels = sorted(df["fault_label"].unique())
    data_by_class = [df.loc[df["fault_label"] == lbl, col].values for lbl in fault_labels]
    bp = axes[i].boxplot(data_by_class, vert=True, patch_artist=True,
                         labels=[l[:15] for l in fault_labels])
    class_colors = sns.color_palette("Set2", len(fault_labels))
    for patch, color in zip(bp["boxes"], class_colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)
    axes[i].set_title(f"{col} by Fault Class")
    axes[i].set_ylabel(col)
    axes[i].tick_params(axis="x", rotation=45)
plt.suptitle("Outlier Analysis - Current Measurements by Fault Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "outlier_current_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'outlier_current_by_class.png'}")
""")

# ===========================================================================
# 1.6 Bivariate Analysis
# ===========================================================================
md("""---
## 1.6 Bivariate Analysis

Study relationships between:
- Ia ↔ Va, Ib ↔ Vb, Ic ↔ Vc (phase pairs)
- Cross-phase measurements by fault class
""")

code("""# Scatter plots: Current vs Voltage for each phase, colored by fault class
phase_pairs = [("Ia", "Va"), ("Ib", "Vb"), ("Ic", "Vc")]
fault_labels = sorted(df["fault_label"].unique())
palette = dict(zip(fault_labels, sns.color_palette("Set2", len(fault_labels))))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, (current, voltage) in enumerate(phase_pairs):
    ax = axes[idx]
    for label in fault_labels:
        mask = df["fault_label"] == label
        ax.scatter(df.loc[mask, current], df.loc[mask, voltage],
                   label=label, alpha=0.5, s=10, color=palette[label], edgecolors="none")
    ax.set_xlabel(current)
    ax.set_ylabel(voltage)
    ax.set_title(f"{current} vs {voltage}")
    if idx == 0:
        ax.legend(fontsize=7, loc="best")
plt.suptitle("Phase Current vs Voltage (colored by fault class)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "bivariate_phase_pairs.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'bivariate_phase_pairs.png'}")
""")

code("""# Grouped box plots: each measurement by fault type
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fault_labels_sorted = sorted(df["fault_label"].unique())
for idx, col in enumerate(measurement_cols):
    ax = axes[idx // 3, idx % 3]
    data_by_class = [df.loc[df["fault_label"] == lbl, col].values for lbl in fault_labels_sorted]
    bp = ax.boxplot(data_by_class, vert=True, patch_artist=True,
                    labels=[l[:12] for l in fault_labels_sorted])
    class_colors = sns.color_palette("Set2", len(fault_labels_sorted))
    for patch, color in zip(bp["boxes"], class_colors):
        patch.set_facecolor(color)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)
    ax.set_title(f"{col} by Fault Class")
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=45)
plt.suptitle("Bivariate Analysis - Measurements by Fault Class", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "bivariate_all_by_class.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'bivariate_all_by_class.png'}")
""")

code("""# Violin plots for current measurements by fault class
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, col in enumerate(["Ia", "Ib", "Ic"]):
    ax = axes[idx]
    fault_labels_sorted = sorted(df["fault_label"].unique())
    data_by_class = [df.loc[df["fault_label"] == lbl, col].values for lbl in fault_labels_sorted]
    vp = ax.violinplot(data_by_class, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(fault_labels_sorted) + 1))
    ax.set_xticklabels([l[:12] for l in fault_labels_sorted], rotation=45, ha="right")
    ax.set_title(f"{col} Distribution by Fault Class")
    ax.set_ylabel(col)
plt.suptitle("Violin Plots - Current by Fault Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "bivariate_violin_current.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'bivariate_violin_current.png'}")
""")

# ===========================================================================
# 1.7 Multivariate / Correlation Analysis
# ===========================================================================
md("""---
## 1.7 Multivariate / Correlation Analysis

Pearson correlation heatmap for raw measurements (Ia, Ib, Ic, Va, Vb, Vc).
**Class labels are NOT included** - they are categorical, not continuous.
""")

code("""# Raw-feature correlation heatmap
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[measurement_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={"label": "Pearson r"}, ax=ax)
ax.set_title("Raw Measurement Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {FIG_DIR / 'correlation_heatmap.png'}")
""")

code("""# Correlation discussion
print("Correlation Analysis Summary:")
print("=" * 70)
print(f"\\nCorrelation matrix (lower triangle):")
for i, col_i in enumerate(measurement_cols):
    for j, col_j in enumerate(measurement_cols):
        if j <= i:
            r = corr.iloc[i, j]
            if i != j:
                strength = "strong" if abs(r) > 0.7 else "moderate" if abs(r) > 0.4 else "weak"
                direction = "positive" if r > 0 else "negative"
                print(f"  {col_i} <-> {col_j}: r = {r:.4f} ({strength} {direction})")

print("\\nKey observations:")
# Find strongest correlations
corr_pairs = []
for i in range(len(measurement_cols)):
    for j in range(i + 1, len(measurement_cols)):
        corr_pairs.append((measurement_cols[i], measurement_cols[j], corr.iloc[i, j]))
corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for col_i, col_j, r in corr_pairs[:5]:
    print(f"  {col_i} <-> {col_j}: r = {r:.4f}")
""")

# ===========================================================================
# Summary
# ===========================================================================
md("""---
## EDA Summary

Key findings from the complete data analysis.
""")

code("""# Build summary dictionary
summary = {
    "dataset": {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "n_classes": len(code_counts),
        "class_distribution": {FAULT_CODE_TO_LABEL.get(k, k): int(v) for k, v in code_counts.items()},
    },
    "data_quality": {
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "infinite_values": int(np.isinf(df[measurement_cols]).sum().sum()),
        "constant_columns": constant_cols,
        "nzv_columns": list(variances[variances < 1e-6].index) if (variances < 1e-6).any() else [],
    },
    "outlier_analysis": {
        col: {
            "n_outliers": int(outlier_summary[col]["n_total"]),
            "pct_outliers": round(outlier_summary[col]["pct"], 2),
            "classification": "physically_meaningful_fault_extremes"
            if (df.loc[(df[col] < outlier_summary[col]["lower_bound"]) |
                       (df[col] > outlier_summary[col]["upper_bound"]), "fault_code"] != "0000").all()
            else "mixed"
        }
        for col in measurement_cols
    },
    "correlation": {
        "strongest_pairs": [
            {"col_i": c[0], "col_j": c[1], "r": round(c[2], 4)}
            for c in corr_pairs[:5]
        ]
    }
}

summary_path = FIG_DIR.parent / "eda_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_path}")
print(f"\\nFigures saved to: {FIG_DIR}")
print(f"Number of figures: {len(list(FIG_DIR.glob('*.png')))}")
""")

# ===========================================================================
# Assemble notebook
# ===========================================================================
nb.cells = cells

# Determine project root
from pathlib import Path
_script_dir = Path(__file__).resolve().parent
ROOT = _script_dir.parents[1]  # results/eda -> results -> root

out_path = ROOT / "results" / "eda" / "eda_report.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook created: {out_path}")
print(f"Total cells: {len(cells)}")
