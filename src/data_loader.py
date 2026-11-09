"""
IG-06 · Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, str(Path(__file__).parent))
from config import PROMPTS_RAW, PROMPTS_REVIEWED, PROCESSED_DIR, LABEL_CATEGORIES, SEED


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSV datasets."""
    df_raw = pd.read_csv(PROMPTS_RAW, encoding="utf-8")
    df_rev = pd.read_csv(PROMPTS_REVIEWED, encoding="utf-8")
    return df_raw, df_rev


def merge_and_enrich(df_raw: pd.DataFrame, df_rev: pd.DataFrame) -> pd.DataFrame:
    """
    Merge datasets: reviewed file is authoritative (has Balochi prompts + human_reviewed=True).
    Fill unreviewed entries from raw file for prompts not in reviewed set.
    """
    # Mark source
    df_rev = df_rev.copy()
    df_rev["dataset_source"] = "reviewed_with_balochi"

    df_raw = df_raw.copy()
    df_raw["dataset_source"] = "synthetic_unreviewed"

    # Reviewed IDs
    reviewed_ids = set(df_rev["prompt_id"])

    # From raw, keep only those NOT already in reviewed
    df_raw_extra = df_raw[~df_raw["prompt_id"].isin(reviewed_ids)].copy()

    # Combine
    df = pd.concat([df_rev, df_raw_extra], ignore_index=True)

    # Clean types
    df["human_reviewed"] = df["human_reviewed"].astype(bool)
    df["has_balochi"] = df["balochi_prompt"].notna()
    df["label"] = pd.Categorical(df["label"], categories=LABEL_CATEGORIES)

    # Prompt length features
    df["prompt_length"] = df["text"].str.len()
    df["prompt_word_count"] = df["text"].str.split().str.len()

    # Label encoding for ML
    label_map = {l: i for i, l in enumerate(LABEL_CATEGORIES)}
    df["label_id"] = df["label"].map(label_map)

    return df.reset_index(drop=True)


def split_by_label(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return dict of label -> subset DataFrame."""
    return {label: df[df["label"] == label].copy() for label in LABEL_CATEGORIES}


def get_paired_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows that have both English and Balochi prompts (for paired analysis)."""
    return df[df["has_balochi"]].copy()


def get_demo_subset(df: pd.DataFrame, n: int = 50, random_state: int = SEED) -> pd.DataFrame:
    """Balanced subset for fast local demos: n // 4 per label."""
    per_label = max(1, n // len(LABEL_CATEGORIES))
    parts = []
    for label in LABEL_CATEGORIES:
        sub = df[df["label"] == label]
        parts.append(sub.sample(min(per_label, len(sub)), random_state=random_state))
    return pd.concat(parts).reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str = "benchmark_combined.csv") -> Path:
    """Save processed DataFrame to processed dir."""
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"Saved → {out}  ({len(df)} rows)")
    return out


if __name__ == "__main__":
    df_raw, df_rev = load_raw_datasets()
    print(f"Raw (unreviewed): {df_raw.shape}")
    print(f"Reviewed (with Balochi): {df_rev.shape}")

    df = merge_and_enrich(df_raw, df_rev)
    print(f"\nMerged dataset: {df.shape}")
    print("Label distribution:")
    print(df["label"].value_counts())
    print(f"\nRows with Balochi prompt: {df['has_balochi'].sum()}")
    print(f"Rows human-reviewed:      {df['human_reviewed'].sum()}")

    save_processed(df)
    print("\n✅ Data preprocessing complete.")
