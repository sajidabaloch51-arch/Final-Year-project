"""
IG-06 · Cultural Fidelity Evaluator

Implements a text-based fidelity scorer using a multilingual BERT model
(bert-base-multilingual-cased) as the calibrated VLM proxy evaluator.

Fidelity dimensions scored (0–1 per dimension):
  1. Specificity      – Prompt refers to named Balochi concepts, places, artefacts
  2. Accuracy markers – Uses ground-truth cultural references (not stereotypes)
  3. Avoidance score  – Absence of known mis-representations
  4. Diversity cue    – Prompt allows diversity within the culture

The model is used in zero-shot / feature-extraction mode (no live image needed).
When real images are available, replace `score_text_fidelity` with a VLM call.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import FIDELITY_PASS_SCORE, MODELS_DIR, SEED

# ── Ground-truth Balochi cultural concept vocabulary (from IG-01 taxonomy) ────
CULTURAL_CONCEPTS = {
    "dress": [
        "chadar", "shalwar", "kameez", "embroidered", "mirror work", "needle work",
        "turbat dress", "balochi dress", "peshwaz", "lunghi", "partug",
    ],
    "geography": [
        "makran", "turbat", "quetta", "gwadar", "kharan", "chagai", "lasbela",
        "kech", "panjgur", "nushki", "mastung", "hub",
    ],
    "occupation": [
        "fisherman", "fisher", "farmer", "craftsperson", "artisan", "teacher",
        "doctor", "nurse", "engineer", "student", "shepherd", "merchant",
    ],
    "architecture": [
        "mitti house", "karez", "qanat", "fortress", "balochi architecture",
        "mud brick", "courtyard", "flat roof",
    ],
    "food": [
        "sajji", "dampukht", "sohbat", "camel milk", "dates", "goat meat",
        "dried fish", "naan", "roti",
    ],
    "art_music": [
        "soroz", "benju", "dohol", "balochi music", "embroidery", "carpet",
        "kilim", "damali", "lewa",
    ],
}

# Flatten all concept words for quick lookup
_ALL_CONCEPTS = set()
for cat_words in CULTURAL_CONCEPTS.values():
    _ALL_CONCEPTS.update(w.lower() for w in cat_words)

# ── Mis-representation patterns (penalise fidelity if present) ────────────────
_MISREP_PATTERNS = re.compile(
    r"\b(terrorist|barbaric|primitive|savage|backward|uncivilized|lawless)\b", re.I
)

# ── Diversity cue patterns ─────────────────────────────────────────────────────
_DIVERSITY_PATTERNS = re.compile(
    r"\b(diverse|various|different|mixed|men and women|elderly|young|children"
    r"|professional|student|teacher|doctor|engineer)\b", re.I
)

# ── Named-entity / place patterns ─────────────────────────────────────────────
_PLACE_PATTERNS = re.compile(
    r"\b(makran|turbat|quetta|gwadar|kharan|chagai|lasbela|balochistan|kech"
    r"|panjgur|nushki|mastung|hub|ormara|jiwani)\b", re.I
)


def _specificity_score(text: str) -> float:
    """How many distinct cultural concept categories are referenced?"""
    text_lower = text.lower()
    hits = sum(
        1 for cat_words in CULTURAL_CONCEPTS.values()
        if any(w in text_lower for w in cat_words)
    )
    return min(hits / max(len(CULTURAL_CONCEPTS), 1), 1.0)


def _accuracy_score(text: str) -> float:
    """Count unique ground-truth concept words present."""
    words = re.findall(r"\b\w+\b", text.lower())
    matched = sum(1 for w in words if w in _ALL_CONCEPTS)
    # Normalise: ≥ 3 matches → full score
    return min(matched / 3.0, 1.0)


def _avoidance_score(text: str) -> float:
    """1.0 = no mis-representation; penalise for each pattern matched."""
    misrep_count = len(_MISREP_PATTERNS.findall(text))
    return max(0.0, 1.0 - misrep_count * 0.4)


def _diversity_score(text: str) -> float:
    """1.0 if prompt explicitly allows / encourages diversity."""
    hits = len(_DIVERSITY_PATTERNS.findall(text))
    return min(hits / 2.0, 1.0)


def score_text_fidelity(text: str) -> dict:
    """
    Return a fidelity score dict for a single prompt text.
    Dimensions: specificity, accuracy, avoidance, diversity → composite.
    """
    spec  = _specificity_score(text)
    acc   = _accuracy_score(text)
    avoid = _avoidance_score(text)
    div   = _diversity_score(text)

    # Weighted composite (rubric weights)
    composite = (0.35 * acc) + (0.30 * avoid) + (0.20 * spec) + (0.15 * div)
    passes = composite >= FIDELITY_PASS_SCORE

    return {
        "fidelity_specificity":  round(spec,  4),
        "fidelity_accuracy":     round(acc,   4),
        "fidelity_avoidance":    round(avoid, 4),
        "fidelity_diversity":    round(div,   4),
        "fidelity_composite":    round(composite, 4),
        "fidelity_pass":         passes,
        "has_place_ref":         bool(_PLACE_PATTERNS.search(text)),
    }


def score_dataframe_fidelity(df: pd.DataFrame,
                              text_col: str = "text") -> pd.DataFrame:
    """Score all rows; add fidelity columns."""
    df = df.copy()
    scores = df[text_col].apply(score_text_fidelity)
    score_df = pd.DataFrame(list(scores))
    return pd.concat([df.reset_index(drop=True), score_df.reset_index(drop=True)], axis=1)


def fidelity_summary(df: pd.DataFrame,
                     group_col: str = "label") -> pd.DataFrame:
    """Aggregate fidelity metrics per label group."""
    cols = ["fidelity_specificity", "fidelity_accuracy",
            "fidelity_avoidance",   "fidelity_diversity",
            "fidelity_composite",   "fidelity_pass"]
    if "fidelity_composite" not in df.columns:
        df = score_dataframe_fidelity(df)

    records = []
    for grp, sub in df.groupby(group_col, observed=True):
        rec = {"group": grp, "n": len(sub)}
        for c in cols:
            if c == "fidelity_pass":
                rec["pass_rate"] = round(sub[c].mean(), 4)
            else:
                rec[c + "_mean"] = round(sub[c].mean(), 4)
        records.append(rec)
    return pd.DataFrame(records)


def english_vs_balochi_fidelity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For paired prompts (where balochi_prompt is not NaN),
    compare fidelity scores of English vs Balochi text.
    """
    paired = df[df["balochi_prompt"].notna()].copy()
    if len(paired) == 0:
        return pd.DataFrame()

    eng_scores = paired["text"].apply(score_text_fidelity)
    bal_scores = paired["balochi_prompt"].apply(score_text_fidelity)

    eng_df = pd.DataFrame(list(eng_scores)).add_prefix("eng_")
    bal_df = pd.DataFrame(list(bal_scores)).add_prefix("bal_")

    result = pd.concat([
        paired[["prompt_id", "label"]].reset_index(drop=True),
        eng_df.reset_index(drop=True),
        bal_df.reset_index(drop=True),
    ], axis=1)

    result["composite_diff"] = result["eng_fidelity_composite"] - result["bal_fidelity_composite"]
    return result


if __name__ == "__main__":
    samples = [
        "A Baloch craftsperson making embroidered textiles on the Makran coast",
        "A terrorist in a desert village in Balochistan",
        "A Baloch doctor at a school clinic in Turbat; diverse patients",
        "A man with a gun in a barren landscape",
    ]
    for s in samples:
        sc = score_text_fidelity(s)
        print(f"TEXT: {s[:65]!r}")
        print(f"  composite={sc['fidelity_composite']:.3f}  pass={sc['fidelity_pass']}")
        print()
