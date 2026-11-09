"""
IG-06 · Calibrated VLM Evaluator
(Text-proxy implementation — replaces VLM when images not available)

Trains a lightweight multilingual text classifier on the benchmark prompts
to predict label category. This is the "calibrated evaluator" spec-required for
scaling beyond human raters.

Pipeline:
  1. Feature extraction from prompt text (TF-IDF + cultural keyword features)
  2. Train a Random Forest classifier on 80/20 split
  3. Report precision/recall/F1 per label
  4. Compute Krippendorff's alpha between classifier and rule-based scores
  5. Save calibration metrics
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (PROCESSED_DIR, MODELS_DIR, LABEL_CATEGORIES,
                    FIDELITY_PASS_SCORE, SEED)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, cohen_kappa_score)
from sklearn.preprocessing import LabelEncoder
import joblib


# ── Krippendorff's alpha (simplified ordinal version) ─────────────────────────
def krippendorff_alpha_nominal(rater1: np.ndarray, rater2: np.ndarray) -> float:
    """
    Simplified Krippendorff's alpha for nominal data (2-rater version).
    Formula:  α = 1 - Do/De
    """
    n = len(rater1)
    if n == 0:
        return 0.0
    # Observed disagreement
    D_o = np.sum(rater1 != rater2) / n
    # Expected disagreement
    vals, counts = np.unique(np.concatenate([rater1, rater2]), return_counts=True)
    freqs = counts / counts.sum()
    D_e = 1.0 - np.sum(freqs ** 2)
    if D_e == 0:
        return 1.0
    return float(1.0 - D_o / D_e)


# ── Feature engineering ───────────────────────────────────────────────────────
from fidelity_evaluator import score_text_fidelity
from stereotype_rubric import detect_attributes


def build_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build hand-crafted features for the calibrated evaluator.
    Combines fidelity scores + stereotype attributes + text stats.
    """
    # Fidelity scores
    fid = df["text"].apply(score_text_fidelity)
    fid_df = pd.DataFrame(list(fid))

    # Stereotype attributes
    attrs = df["text"].apply(detect_attributes)
    attr_df = pd.DataFrame(list(attrs)).astype(int)

    # Basic text features
    text_feats = pd.DataFrame({
        "prompt_len":    df["text"].str.len(),
        "word_count":    df["text"].str.split().str.len(),
        "has_balochi":   df["balochi_prompt"].notna().astype(int),
    })

    return pd.concat([
        fid_df.reset_index(drop=True),
        attr_df.reset_index(drop=True),
        text_feats.reset_index(drop=True),
    ], axis=1)


def train_calibrated_evaluator(df: pd.DataFrame) -> dict:
    """
    Train + evaluate the calibrated VLM proxy classifier.

    Returns metrics dict and saves model to outputs/models/.
    """
    print("── Training Calibrated VLM Evaluator ──────────────────────────────")

    # Build features
    print("  Building features …")
    feat_df = build_feature_df(df)

    # TF-IDF on raw text
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                             sublinear_tf=True, strip_accents="unicode")
    X_tfidf = tfidf.fit_transform(df["text"].fillna("")).toarray()

    X_hand  = feat_df.select_dtypes(include=[np.number]).fillna(0).values
    X       = np.hstack([X_tfidf, X_hand])

    le  = LabelEncoder()
    y   = le.fit_transform(df["label"].astype(str))

    # Train/test split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y)

    # ── Three classifiers ──────────────────────────────────────────────────────
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=500, C=1.0, random_state=SEED, solver="lbfgs"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=SEED),
    }

    results = {}
    best_name, best_acc = None, 0.0

    for name, clf in models.items():
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        kappa  = cohen_kappa_score(y_te, y_pred)
        report = classification_report(
            y_te, y_pred, target_names=le.classes_, output_dict=True)

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(clf, X_tr, y_tr, cv=cv, scoring="accuracy")

        results[name] = {
            "accuracy":       round(acc,   4),
            "kappa":          round(kappa, 4),
            "cv_mean":        round(cv_scores.mean(), 4),
            "cv_std":         round(cv_scores.std(),  4),
            "report":         report,
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        }
        print(f"  {name:<22}  acc={acc:.3f}  kappa={kappa:.3f}  "
              f"cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_clf  = clf

    # ── Krippendorff's alpha (rule-based vs best classifier) ──────────────────
    y_rule  = df["label_id"].values if "label_id" in df.columns else y
    y_clf   = best_clf.predict(X)
    kalpha  = krippendorff_alpha_nominal(y_rule[:len(y_clf)], y_clf)
    print(f"\n  Best model: {best_name} (acc={best_acc:.3f})")
    print(f"  Krippendorff's α (rule vs classifier): {kalpha:.4f}")

    # ── Save best model ────────────────────────────────────────────────────────
    model_path = MODELS_DIR / "calibrated_evaluator.pkl"
    joblib.dump({"model": best_clf, "tfidf": tfidf, "le": le,
                 "feat_cols": list(feat_df.select_dtypes(include=[np.number]).columns)},
                model_path)
    print(f"  Model saved → {model_path}")

    # Save metrics JSON
    metrics = {
        "best_model":          best_name,
        "best_accuracy":       round(best_acc, 4),
        "krippendorff_alpha":  round(kalpha, 4),
        "model_results":       {k: {ek: ev for ek, ev in v.items()
                                    if ek != "report"}
                                for k, v in results.items()},
        "label_classes":       list(le.classes_),
    }
    metrics_path = PROCESSED_DIR / "evaluator_calibration_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    return {**metrics, "all_results": results,
            "le": le, "best_clf": best_clf, "tfidf": tfidf}


def load_evaluator():
    """Load the saved calibrated evaluator."""
    model_path = MODELS_DIR / "calibrated_evaluator.pkl"
    return joblib.load(model_path)


if __name__ == "__main__":
    from data_loader import load_raw_datasets, merge_and_enrich
    from fidelity_evaluator import score_dataframe_fidelity
    from stereotype_rubric import score_dataframe as stereotype_score

    df_raw, df_rev = load_raw_datasets()
    df = merge_and_enrich(df_raw, df_rev)
    df = score_dataframe_fidelity(df)
    df = stereotype_score(df)
    metrics = train_calibrated_evaluator(df)
    print("\n✅ Calibrated evaluator training complete.")
