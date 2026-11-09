"""
IG-06 · Extended Visualisations
Figures 9–16 covering model comparison and calibrated evaluator results.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from config import FIGURES_DIR, LABEL_CATEGORIES, STEREOTYPE_ALERT, FIDELITY_PASS_SCORE

sns.set_theme(style="whitegrid", palette="muted")

MODEL_COLORS = {
    "DALL-E (OpenAI)":         "#4C72B0",
    "Stable Diffusion (open)": "#DD8452",
    "Midjourney (commercial)": "#55A868",
    "Imagen (Google)":         "#C44E52",
}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 · Model fidelity comparison (bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_fidelity(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors  = [MODEL_COLORS.get(m, "#999") for m in summary["model"]]
    bars    = ax.bar(summary["model"], summary["fidelity_mean"],
                     color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, fontweight="bold")
    ax.axhline(FIDELITY_PASS_SCORE, color="crimson", linestyle="--",
               linewidth=1.3, label=f"Pass threshold ({FIDELITY_PASS_SCORE})", zorder=2)
    ax.set_ylim(0, max(summary["fidelity_mean"]) * 1.25)
    ax.set_title("Mean Cultural Fidelity Score by Generator Model",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean Fidelity (0–1)")
    ax.legend(fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "fig9_model_fidelity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 · Stereotype rates by model (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_stereotypes(summary: pd.DataFrame) -> Path:
    attrs = ["desert_arid_rate", "weapons_present_rate",
             "poverty_cues_rate", "male_only_rate"]
    nice  = ["Desert/Arid", "Weapons", "Poverty Cues", "Male Only"]

    x = np.arange(len(summary))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (col, label) in enumerate(zip(attrs, nice)):
        if col in summary.columns:
            offset = (i - 1.5) * width
            ax.bar(x + offset, summary[col], width, label=label,
                   alpha=0.88, edgecolor="white", linewidth=0.5)

    ax.axhline(STEREOTYPE_ALERT, color="crimson", linestyle="--",
               linewidth=1.2, label=f"Alert threshold ({STEREOTYPE_ALERT:.0%})")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stereotype Default Rates by Generator Model",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Default Rate (0–1)")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig10_model_stereotypes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 · Violence in neutral prompts — per model
# ─────────────────────────────────────────────────────────────────────────────
def plot_violence_per_model(summary: pd.DataFrame) -> Path:
    if "violence_neutral_rate" not in summary.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#e07b54" if v >= STEREOTYPE_ALERT else "#4C72B0"
               for v in summary["violence_neutral_rate"]]
    bars    = ax.bar(summary["model"], summary["violence_neutral_rate"],
                     color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=10, fontweight="bold")
    ax.axhline(STEREOTYPE_ALERT, color="crimson", linestyle="--",
               linewidth=1.3, label=f"Alert threshold ({STEREOTYPE_ALERT:.0%})", zorder=2)
    ax.set_ylim(0, max(summary["violence_neutral_rate"]) * 1.3 + 0.05)
    ax.set_title("Violence/Weapons in Neutral Prompts by Model",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Violence Rate")
    ax.legend(fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "fig11_violence_per_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 · Fidelity heatmap: model × label
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_label_heatmap(per_label: pd.DataFrame) -> Path:
    pivot = per_label.pivot(index="model", columns="label", values="fidelity_mean")
    pivot = pivot.reindex(columns=LABEL_CATEGORIES, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn",
                vmin=0, vmax=1, linewidths=0.5, linecolor="white",
                ax=ax, cbar_kws={"label": "Fidelity"})
    ax.set_title("Fidelity Score: Model × Label Category",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Label Category")
    ax.set_ylabel("Model")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / "fig12_model_label_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13 · Fidelity distribution violin by model
# ─────────────────────────────────────────────────────────────────────────────
def plot_fidelity_violin(sim_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    models = list(MODEL_COLORS.keys())
    data   = [sim_df[sim_df["model"] == m]["fidelity_score"].values for m in models]
    parts  = ax.violinplot(data, positions=range(len(models)),
                           showmedians=True, showextrema=True)
    for i, (pc, m) in enumerate(zip(parts["bodies"], models)):
        pc.set_facecolor(MODEL_COLORS.get(m, "#999"))
        pc.set_alpha(0.75)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.axhline(FIDELITY_PASS_SCORE, color="crimson", linestyle="--",
               linewidth=1.3, label=f"Pass threshold ({FIDELITY_PASS_SCORE})")
    ax.set_ylim(0, 1.05)
    ax.set_title("Fidelity Score Distribution by Model (per-image)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Fidelity Score")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig13_fidelity_violin.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14 · Evaluator confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm_data: list, label_names: list) -> Path:
    import numpy as np
    cm = np.array(cm_data)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Row-normalised Rate"})
    ax.set_title("Calibrated Evaluator Confusion Matrix\n(row-normalised)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = FIGURES_DIR / "fig14_confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 15 · Per-model gender default rates
# ─────────────────────────────────────────────────────────────────────────────
def plot_gender_per_model(sim_df: pd.DataFrame) -> Path:
    neutral = sim_df[sim_df["label"] == "neutral_control"]
    if len(neutral) == 0:
        return None

    gender_data = neutral.groupby("model")["male_only"].mean().reset_index()
    gender_data.columns = ["model", "male_only_rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = [MODEL_COLORS.get(m, "#999") for m in gender_data["model"]]
    bars    = ax.bar(gender_data["model"], gender_data["male_only_rate"],
                     color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar_label(bars, fmt="%.1%%", padding=4, fontsize=10, fontweight="bold")
    ax.axhline(STEREOTYPE_ALERT, color="crimson", linestyle="--",
               linewidth=1.3, label=f"Alert ({STEREOTYPE_ALERT:.0%})", zorder=2)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, label="50/50 baseline")
    ax.set_ylim(0, 1.05)
    ax.set_title("Male-only Default Rate in Neutral Prompts by Model",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Male-only Rate")
    ax.legend(fontsize=9)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out = FIGURES_DIR / "fig15_gender_per_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 16 · Radar chart — model stereotype profile
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_model_profile(summary: pd.DataFrame) -> Path:
    categories = ["Desert/Arid", "Weapons", "Poverty", "Male Only", "Conflict"]
    cols       = ["desert_arid_rate", "weapons_present_rate",
                  "poverty_cues_rate", "male_only_rate", "conflict_cues_rate"]
    cols_avail = [c for c in cols if c in summary.columns]
    cats_avail = [categories[i] for i, c in enumerate(cols) if c in summary.columns]

    N    = len(cats_avail)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats_avail, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)

    # Alert circle
    circle_angles = np.linspace(0, 2 * np.pi, 200)
    ax.plot(circle_angles, [STEREOTYPE_ALERT] * 200, "r--", linewidth=1, alpha=0.5)

    for _, row in summary.iterrows():
        values = [float(row.get(c, 0)) for c in cols_avail]
        values += values[:1]
        color = MODEL_COLORS.get(row["model"], "#999")
        ax.plot(angles, values, linewidth=2, linestyle="solid", color=color,
                label=row["model"])
        ax.fill(angles, values, alpha=0.10, color=color)

    ax.set_title("Stereotype Profile Radar by Model\n(red dashed = 50% alert)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "fig16_radar_model_profile.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {out.name}")
    return out


def generate_extended_figures(model_results: dict, evaluator_metrics: dict = None):
    """Generate Figs 9–16."""
    print("\n── Generating extended figures (Figs 9–16) ──")
    sim_df  = model_results["sim_df"]
    summary = model_results["summary"]
    per_lab = model_results["per_label"]

    plot_model_fidelity(summary)
    plot_model_stereotypes(summary)
    plot_violence_per_model(summary)
    plot_model_label_heatmap(per_lab)
    plot_fidelity_violin(sim_df)
    plot_gender_per_model(sim_df)
    plot_radar_model_profile(summary)

    if evaluator_metrics:
        best_model_name = evaluator_metrics["best_model"]
        all_results     = evaluator_metrics.get("all_results", {})
        if best_model_name in all_results:
            cm   = all_results[best_model_name]["confusion_matrix"]
            lbls = evaluator_metrics["label_classes"]
            plot_confusion_matrix(cm, lbls)

    print(f"  Extended figures saved to: {FIGURES_DIR}")
