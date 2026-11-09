"""
IG-06 · Visualisation module
Generates all benchmark figures and saves to outputs/figures/
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES_DIR, LABEL_CATEGORIES, STEREOTYPE_ALERT, FIDELITY_PASS_SCORE

sns.set_theme(style="whitegrid", palette="muted")
COLORS = {
    "neutral_control":      "#4C72B0",
    "gender_representation":"#DD8452",
    "cultural_fidelity":    "#55A868",
    "stereotype_audit":     "#C44E52",
}


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 · Label distribution
# ─────────────────────────────────────────────────────────────────────────────
def plot_label_distribution(df: pd.DataFrame) -> Path:
    counts = df["label"].value_counts().reindex(LABEL_CATEGORIES)
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=[COLORS[l] for l in counts.index], edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=11, fontweight="bold")
    ax.set_title("Prompt Label Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Number of Prompts")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_label_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 · Fidelity scores by label
# ─────────────────────────────────────────────────────────────────────────────
def plot_fidelity_by_label(fid_summary: pd.DataFrame) -> Path:
    dims = ["fidelity_accuracy_mean", "fidelity_specificity_mean",
            "fidelity_avoidance_mean", "fidelity_diversity_mean"]
    labels_nice = ["Accuracy", "Specificity", "Avoidance", "Diversity"]

    x = np.arange(len(fid_summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (dim, nice) in enumerate(zip(dims, labels_nice)):
        if dim in fid_summary.columns:
            offset = (i - 1.5) * width
            ax.bar(x + offset, fid_summary[dim], width,
                   label=nice, alpha=0.88, edgecolor="white", linewidth=0.6)

    ax.axhline(FIDELITY_PASS_SCORE, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Pass threshold ({FIDELITY_PASS_SCORE})")
    ax.set_xticks(x)
    ax.set_xticklabels(fid_summary["group"], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cultural Fidelity Scores by Label Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score (0–1)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_fidelity_by_label.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 · Stereotype default rates heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_stereotype_heatmap(default_rates: pd.DataFrame) -> Path:
    pivot = default_rates.pivot(index="attribute", columns="group", values="rate")
    pivot = pivot.reindex(columns=LABEL_CATEGORIES, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Default Rate"})

    # Overlay red borders on alerts
    for y_idx, attr in enumerate(pivot.index):
        for x_idx, grp in enumerate(pivot.columns):
            rate = pivot.loc[attr, grp]
            if rate >= STEREOTYPE_ALERT:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (x_idx, y_idx), 1, 1,
                    boxstyle="square,pad=0", linewidth=2,
                    edgecolor="crimson", facecolor="none", transform=ax.transData
                ))

    ax.set_title("Stereotype Default Rates by Attribute & Label\n"
                 "(Red border = alert ≥ 50 %)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Stereotype Attribute")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_stereotype_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 · Violence/Weapons headline metric
# ─────────────────────────────────────────────────────────────────────────────
def plot_violence_headline(violence_metric: dict) -> Path:
    labels = ["Weapons\nRate", "Conflict\nRate", "Either\nViolence Rate"]
    values = [violence_metric["weapons_rate"],
              violence_metric["conflict_rate"],
              violence_metric["either_violence_rate"]]
    colors_bar = ["#e07b54" if v >= STEREOTYPE_ALERT else "#4C72B0" for v in values]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=11, fontweight="bold")
    ax.axhline(STEREOTYPE_ALERT, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Alert threshold ({STEREOTYPE_ALERT:.0%})")
    ax.set_ylim(0, max(values) * 1.3 + 0.05)
    ax.set_title(f"Violence/Weapons Cues in Neutral Prompts\n"
                 f"(n={violence_metric['neutral_prompt_count']})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Prompt Rate")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_violence_headline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 · Gender defaults
# ─────────────────────────────────────────────────────────────────────────────
def plot_gender_defaults(gender_rates: pd.DataFrame) -> Path:
    if gender_rates.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = gender_rates.pivot(index="group", columns="attribute", values="rate")
    pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"],
               edgecolor="white", linewidth=0.8)
    ax.axhline(STEREOTYPE_ALERT, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Alert threshold ({STEREOTYPE_ALERT:.0%})")
    ax.set_title("Gender Default Rates by Label\n(neutral_control & gender_representation)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Prompt Rate")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    plt.xticks(rotation=10)
    plt.tight_layout()
    out = FIGURES_DIR / "fig5_gender_defaults.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 · English vs Balochi fidelity comparison
# ─────────────────────────────────────────────────────────────────────────────
def plot_eng_vs_balochi(paired_df: pd.DataFrame) -> Path:
    if paired_df is None or len(paired_df) == 0:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter
    ax = axes[0]
    ax.scatter(paired_df["eng_fidelity_composite"],
               paired_df["bal_fidelity_composite"],
               alpha=0.4, s=18, color="#4C72B0", edgecolors="none")
    lim = [0, paired_df[["eng_fidelity_composite","bal_fidelity_composite"]].max().max() + 0.05]
    ax.plot(lim, lim, "k--", linewidth=1, label="Equal line")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("English Prompt Fidelity"); ax.set_ylabel("Balochi Prompt Fidelity")
    ax.set_title("Fidelity: English vs Balochi Prompts", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)

    # Distribution of diff
    ax2 = axes[1]
    ax2.hist(paired_df["composite_diff"], bins=30, color="#55A868", edgecolor="white",
             linewidth=0.6, alpha=0.85)
    ax2.axvline(0, color="crimson", linestyle="--", linewidth=1.2)
    mean_diff = paired_df["composite_diff"].mean()
    ax2.axvline(mean_diff, color="#4C72B0", linestyle="-", linewidth=1.4,
                label=f"Mean diff = {mean_diff:+.3f}")
    ax2.set_title("Fidelity Difference (English − Balochi)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fidelity Difference")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = FIGURES_DIR / "fig6_eng_vs_balochi.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 · Composite fidelity KDE by label
# ─────────────────────────────────────────────────────────────────────────────
def plot_fidelity_kde(df: pd.DataFrame) -> Path:
    if "fidelity_composite" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    for label in LABEL_CATEGORIES:
        sub = df[df["label"] == label]["fidelity_composite"].dropna()
        sub.plot.kde(ax=ax, label=label, color=COLORS.get(label), linewidth=2)
    ax.axvline(FIDELITY_PASS_SCORE, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Pass threshold ({FIDELITY_PASS_SCORE})")
    ax.set_xlim(0, 1)
    ax.set_title("Fidelity Score Distribution by Label (KDE)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Composite Fidelity Score")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig7_fidelity_kde.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 · Pass rate by label (bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_pass_rate(fid_summary: pd.DataFrame) -> Path:
    if "pass_rate" not in fid_summary.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(fid_summary["group"], fid_summary["pass_rate"],
                  color=[COLORS.get(g, "#999") for g in fid_summary["group"]],
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=11, fontweight="bold")
    ax.axhline(FIDELITY_PASS_SCORE, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Pass threshold ({FIDELITY_PASS_SCORE})")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fidelity Pass Rate by Label Category",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Pass Rate")
    ax.legend(fontsize=9)
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "fig8_pass_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")
    return out


def generate_all_figures(results: dict):
    """Generate all 8 figures from pipeline results."""
    print("\n── Generating figures ──")
    df         = results["df_scored"]
    fid_sum    = results["fidelity_summary"]
    def_rates  = results["default_rates"]
    gender_r   = results["gender_rates"]
    violence_m = results["violence_metric"]
    paired     = results["paired_df"]

    plot_label_distribution(df)
    plot_fidelity_by_label(fid_sum)
    plot_stereotype_heatmap(def_rates)
    plot_violence_headline(violence_m)
    plot_gender_defaults(gender_r)
    plot_eng_vs_balochi(paired)
    plot_fidelity_kde(df)
    plot_pass_rate(fid_sum)
    print(f"\n  All figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import run_pipeline
    results = run_pipeline(demo=False)
    generate_all_figures(results)
