"""Calibrate the NLP decision threshold on the held-out 20% split.

Sweeps probability thresholds [0.05 .. 0.95] in 0.01 steps and picks the value
that maximises **macro-F1** by default; ``--metric youden`` picks the Youden's
J point on the ROC instead.

Writes ``models/nlp_threshold.json`` containing:
    {
      "threshold": 0.42,
      "metric": "macro_f1",
      "macro_f1_at_threshold": ...,
      "macro_f1_at_0.5": ...,
      ...
    }

The trained weights are NOT modified. ``NLPAgent`` reads this file at startup
to override the default 0.5 cutoff.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    MODELS_DIR,
    NLP_MODEL_PATH,
    NLP_VECTORIZER_PATH,
    NORMAL_LABEL,
    TEXT_CSV,
)
from text_clean import prepare_dataframe  # noqa: E402


def calibrate(*, metric: str) -> dict:
    if not Path(TEXT_CSV).is_file():
        raise SystemExit(f"Dataset not found: {TEXT_CSV}")
    if not Path(NLP_MODEL_PATH).is_file() or not Path(NLP_VECTORIZER_PATH).is_file():
        raise SystemExit("Trained NLP model / vectorizer missing.")

    df = pd.read_csv(TEXT_CSV)
    df = prepare_dataframe(df, "statement")
    y = (df["status"] != NORMAL_LABEL).astype(int).to_numpy()
    X = df["clean_text"].to_numpy()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    vec = joblib.load(NLP_VECTORIZER_PATH)
    clf = joblib.load(NLP_MODEL_PATH)
    Xv = vec.transform(X_test)
    classes = list(clf.classes_)
    idx_neg = classes.index(1) if 1 in classes else 1
    p_neg = clf.predict_proba(Xv)[:, idx_neg]

    base = (p_neg >= 0.5).astype(int)
    base_macro = float(f1_score(y_test, base, average="macro", zero_division=0))

    if metric == "youden":
        fpr, tpr, thr = roc_curve(y_test, p_neg)
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        best_thr = float(np.clip(thr[best_idx], 0.0, 1.0))
        best_macro = float(
            f1_score(y_test, (p_neg >= best_thr).astype(int), average="macro", zero_division=0)
        )
    else:
        thresholds = np.linspace(0.05, 0.95, 91)
        best_thr, best_macro = 0.5, base_macro
        for t in thresholds:
            preds = (p_neg >= float(t)).astype(int)
            f = float(f1_score(y_test, preds, average="macro", zero_division=0))
            if f > best_macro:
                best_thr, best_macro = float(t), f

    out = {
        "threshold": float(round(best_thr, 4)),
        "metric": metric,
        "macro_f1_at_threshold": best_macro,
        "macro_f1_at_0.5": base_macro,
        "delta_macro_f1": best_macro - base_macro,
        "n_test": int(len(y_test)),
    }
    out_path = MODELS_DIR / "nlp_threshold.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"[calibrate] wrote {out_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metric",
        choices=["macro_f1", "youden"],
        default="macro_f1",
        help="Optimisation criterion (default: macro_f1).",
    )
    args = ap.parse_args()
    calibrate(metric=args.metric)


if __name__ == "__main__":
    main()
