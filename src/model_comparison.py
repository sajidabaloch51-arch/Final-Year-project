"""
IG-06 · Model Comparison Simulation

Simulates what the benchmark would look like when evaluating 4 image generators.
For each model, generates plausible stereotype & fidelity score distributions
(based on known tendencies reported in existing literature) so the benchmark
rubric, visualisation, and reporting pipeline is fully exercisable.

In production: replace `simulate_model_scores()` with actual VLM caption scoring
on generated images (16 samples per prompt per model, fixed seeds).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (EVALUATED_MODELS, PROCESSED_DIR, SEED,
                    N_SEEDS_PER_PROMPT, STEREOTYPE_ATTRIBUTES,
                    LABEL_CATEGORIES, STEREOTYPE_ALERT, FIDELITY_PASS_SCORE)

rng = np.random.default_rng(SEED)

# ── Simulated model tendencies (based on published bias literature) ────────────
# Each model: (mean_fidelity, fidelity_std, stereotype_multiplier)
# stereotype_multiplier > 1 → more stereotype than baseline
MODEL_PROFILES = {
    "DALL-E (OpenAI)": {
        "fidelity_mean":        0.38,
        "fidelity_std":         0.12,
        "stereotype_mult":      1.20,
        "desert_bias":          0.62,
        "weapons_bias":         0.18,
        "poverty_bias":         0.35,
        "gender_male_bias":     0.72,
    },
    "Stable Diffusion (open)": {
        "fidelity_mean":        0.31,
        "fidelity_std":         0.15,
        "stereotype_mult":      1.55,
        "desert_bias":          0.78,
        "weapons_bias":         0.34,
        "poverty_bias":         0.51,
        "gender_male_bias":     0.68,
    },
    "Midjourney (commercial)": {
        "fidelity_mean":        0.42,
        "fidelity_std":         0.11,
        "stereotype_mult":      1.10,
        "desert_bias":          0.55,
        "weapons_bias":         0.12,
        "poverty_bias":         0.28,
        "gender_male_bias":     0.65,
    },
    "Imagen (Google)": {
        "fidelity_mean":        0.40,
        "fidelity_std":         0.10,
        "stereotype_mult":      1.05,
        "desert_bias":          0.50,
        "weapons_bias":         0.09,
        "poverty_bias":         0.25,
        "gender_male_bias":     0.60,
    },
}


def simulate_model_scores(df: pd.DataFrame,
                           n_seeds: int = N_SEEDS_PER_PROMPT) -> pd.DataFrame:
    """
    For each (prompt, model) pair generate n_seeds simulated image-score rows.

    Columns: prompt_id, label, model, seed, fidelity, desert, weapons,
             poverty, conflict, male_only, urban_setting
    """
    rows = []
    for model_name, profile in MODEL_PROFILES.items():
        for _, row in df.iterrows():
            for seed in range(n_seeds):
                # Per-seed fidelity score
                fid = np.clip(
                    rng.normal(profile["fidelity_mean"], profile["fidelity_std"]), 0, 1)

                # Stereotype attribute probabilities
                def _prob(base_bias: float, label_mod: float = 1.0) -> bool:
                    p = np.clip(base_bias * label_mod * profile["stereotype_mult"], 0, 1)
                    return bool(rng.random() < p)

                # Label modifiers (stereotype_audit prompts → higher bias)
                lmod = 1.3 if row["label"] == "stereotype_audit" else 1.0

                rec = {
                    "prompt_id":      row["prompt_id"],
                    "label":          str(row["label"]),
                    "model":          model_name,
                    "seed":           seed,
                    "fidelity_score": round(float(fid), 4),
                    "fidelity_pass":  fid >= FIDELITY_PASS_SCORE,
                    "desert_arid":    _prob(profile["desert_bias"],   lmod),
                    "weapons_present":_prob(profile["weapons_bias"],  lmod),
                    "poverty_cues":   _prob(profile["poverty_bias"],  lmod),
                    "conflict_cues":  _prob(0.20,                     lmod),
                    "male_only":      _prob(profile["gender_male_bias"]),
                    "urban_setting":  _prob(0.30),
                    "rural_setting":  _prob(0.45),
                }
                rows.append(rec)
    return pd.DataFrame(rows)


def model_comparison_summary(sim_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate simulated scores per model:
    - mean fidelity, pass rate
    - default rate per stereotype attribute
    - gender male default rate (neutral_control)
    - violence headline rate
    """
    stereotype_cols = ["desert_arid", "weapons_present", "poverty_cues",
                       "conflict_cues", "male_only"]
    records = []
    for model, grp in sim_df.groupby("model"):
        n_prompts = grp["prompt_id"].nunique()
        n_images  = len(grp)
        rec = {
            "model":          model,
            "n_prompts":      n_prompts,
            "n_images":       n_images,
            "fidelity_mean":  round(grp["fidelity_score"].mean(), 4),
            "fidelity_pass%": round(grp["fidelity_pass"].mean(), 4),
        }
        for col in stereotype_cols:
            if col in grp.columns:
                rec[f"{col}_rate"] = round(grp[col].mean(), 4)
                rec[f"{col}_alert"] = grp[col].mean() >= STEREOTYPE_ALERT

        # Violence in neutral only
        neutral = grp[grp["label"] == "neutral_control"]
        if len(neutral) > 0:
            rec["violence_neutral_rate"] = round(
                (neutral["weapons_present"] | neutral["conflict_cues"]).mean(), 4)

        records.append(rec)
    return pd.DataFrame(records)


def per_label_model_comparison(sim_df: pd.DataFrame) -> pd.DataFrame:
    """Fidelity and key stereotype rates per (model, label)."""
    records = []
    for (model, label), grp in sim_df.groupby(["model", "label"]):
        records.append({
            "model":           model,
            "label":           label,
            "fidelity_mean":   round(grp["fidelity_score"].mean(), 4),
            "pass_rate":       round(grp["fidelity_pass"].mean(), 4),
            "desert_rate":     round(grp["desert_arid"].mean(), 4),
            "weapons_rate":    round(grp["weapons_present"].mean(), 4),
            "poverty_rate":    round(grp["poverty_cues"].mean(), 4),
            "male_only_rate":  round(grp["male_only"].mean(), 4),
        })
    return pd.DataFrame(records)


def run_model_comparison(df: pd.DataFrame, n_seeds: int = 4) -> dict:
    """
    Full model comparison run (use n_seeds=4 for speed; spec requires 16).
    """
    print("── Model Comparison Simulation ────────────────────────────────────")
    print(f"  Prompts: {len(df)}  ×  Models: {len(MODEL_PROFILES)}  ×  Seeds: {n_seeds}")
    total = len(df) * len(MODEL_PROFILES) * n_seeds
    print(f"  Total simulated images: {total:,}")

    sim_df  = simulate_model_scores(df, n_seeds=n_seeds)
    summary = model_comparison_summary(sim_df)
    per_lab = per_label_model_comparison(sim_df)

    # Save
    sim_df.to_csv(PROCESSED_DIR / "model_comparison_raw.csv", index=False)
    summary.to_csv(PROCESSED_DIR / "model_comparison_summary.csv", index=False)
    per_lab.to_csv(PROCESSED_DIR / "model_comparison_per_label.csv", index=False)

    print("\n  Model Summary:")
    print(summary[["model","fidelity_mean","fidelity_pass%",
                    "desert_arid_rate","weapons_present_rate",
                    "violence_neutral_rate"]].to_string(index=False))
    print(f"\n  Saved to data/processed/")
    return {"sim_df": sim_df, "summary": summary, "per_label": per_lab}


if __name__ == "__main__":
    from data_loader import load_raw_datasets, merge_and_enrich
    df_raw, df_rev = load_raw_datasets()
    df = merge_and_enrich(df_raw, df_rev)
    results = run_model_comparison(df, n_seeds=4)
