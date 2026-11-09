"""
IG-06 · Evaluation Pipeline

Main evaluation runner:
  1. Load & merge datasets
  2. Score fidelity
  3. Score stereotypes
  4. Compute headline metrics (violence/weapons, gender defaults)
  5. Compare English vs Balochi prompts
  6. Save all results to outputs/
"""
import sys, os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DIR, FIGURES_DIR, OUTPUTS_DIR,
                    STEREOTYPE_ALERT, FIDELITY_PASS_SCORE)
from data_loader import (load_raw_datasets, merge_and_enrich,
                         get_paired_prompts, get_demo_subset, save_processed)
from fidelity_evaluator import (score_dataframe_fidelity, fidelity_summary,
                                 english_vs_balochi_fidelity)
from stereotype_rubric import (score_dataframe as stereotype_score,
                                compute_default_rates,
                                compute_gender_default_rates,
                                violence_headline_metric)


def run_pipeline(demo: bool = False, n_demo: int = 200) -> dict:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    demo : bool
        If True, run on a balanced subset of n_demo rows.
    n_demo : int
        Size of demo subset.

    Returns
    -------
    dict with all result DataFrames and metrics.
    """
    print("=" * 60)
    print("IG-06 · Balochistan T2I Benchmark · Evaluation Pipeline")
    print("=" * 60)

    # 1. Load ─────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets …")
    df_raw, df_rev = load_raw_datasets()
    print(f"  Unreviewed (English only): {len(df_raw):,} rows")
    print(f"  Reviewed (with Balochi):   {len(df_rev):,} rows")

    df = merge_and_enrich(df_raw, df_rev)
    print(f"  Merged:                    {len(df):,} rows")

    if demo:
        df = get_demo_subset(df, n=n_demo)
        print(f"  Demo subset:               {len(df):,} rows")

    # 2. Fidelity scoring ─────────────────────────────────────────────────────
    print("\n[2/6] Scoring cultural fidelity …")
    df = score_dataframe_fidelity(df)
    pass_rate = df["fidelity_pass"].mean()
    print(f"  Overall fidelity pass rate: {pass_rate:.1%}")

    # 3. Stereotype scoring ───────────────────────────────────────────────────
    print("\n[3/6] Scoring stereotype attributes …")
    df = stereotype_score(df)

    # 4. Headline metrics ─────────────────────────────────────────────────────
    print("\n[4/6] Computing headline metrics …")
    violence_metric = violence_headline_metric(df)
    print(f"  Neutral prompts: {violence_metric['neutral_prompt_count']}")
    print(f"  Weapons/violence rate: {violence_metric['either_violence_rate']:.1%}  "
          f"{'⚠️  ALERT' if violence_metric['alert'] else '✅ OK'}")

    default_rates = compute_default_rates(df)
    gender_rates  = compute_gender_default_rates(df)

    # 5. English vs Balochi comparison ────────────────────────────────────────
    print("\n[5/6] Comparing English vs Balochi prompts …")
    paired_df = english_vs_balochi_fidelity(df)
    if len(paired_df) > 0:
        mean_diff = paired_df["composite_diff"].mean()
        print(f"  Paired prompts: {len(paired_df):,}")
        print(f"  Mean fidelity diff (ENG - BAL): {mean_diff:+.4f}")
    else:
        print("  No paired prompts in current subset.")
        mean_diff = None

    # 6. Aggregate summaries ──────────────────────────────────────────────────
    print("\n[6/6] Building aggregate summaries …")
    fid_summary = fidelity_summary(df)
    print("\n  Fidelity summary by label:")
    print(fid_summary.to_string(index=False))

    stereotype_alerts = default_rates[default_rates["alert"]]
    print(f"\n  Stereotype alert attributes: {len(stereotype_alerts)}")
    if len(stereotype_alerts) > 0:
        print(stereotype_alerts[["group","attribute","rate"]].to_string(index=False))

    # ── Save all outputs ──────────────────────────────────────────────────────
    save_processed(df, "benchmark_scored.csv")
    default_rates.to_csv(PROCESSED_DIR / "stereotype_default_rates.csv", index=False)
    gender_rates.to_csv(PROCESSED_DIR / "gender_default_rates.csv", index=False)
    fid_summary.to_csv(PROCESSED_DIR / "fidelity_summary.csv", index=False)
    if len(paired_df) > 0:
        paired_df.to_csv(PROCESSED_DIR / "english_vs_balochi.csv", index=False)

    print("\n✅ Evaluation pipeline complete. Results saved to data/processed/")

    return {
        "df_scored":        df,
        "fidelity_summary": fid_summary,
        "default_rates":    default_rates,
        "gender_rates":     gender_rates,
        "violence_metric":  violence_metric,
        "paired_df":        paired_df,
        "overall_pass_rate": pass_rate,
        "eng_bal_diff":     mean_diff,
    }


if __name__ == "__main__":
    results = run_pipeline(demo=False)
