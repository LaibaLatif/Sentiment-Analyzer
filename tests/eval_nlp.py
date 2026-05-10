"""Evaluate the trained NLP head on a held-out split of ``Combined Data.csv``.

Writes ``models/nlp_eval.json`` (per-class precision / recall / F1 / support,
ROC-AUC, PR-AUC, accuracy, macro / weighted F1) and ``models/nlp_confusion.png``.

The split is reproduced with the same ``test_size=0.2 / random_state=42`` used
during training — this matches what the model has not seen.

Run from project root:
    python -m scripts.eval_nlp
    python -m scripts.eval_nlp --threshold 0.45
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
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


def _save_confusion_png(cm: np.ndarray, classes: list[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=120)
    im = ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=15, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("NLP — confusion matrix (held-out 20% split)", fontsize=10)
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=10,
                color="white" if cm[i, j] > thresh else "#1e293b",
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate(*, threshold: float | None) -> dict:
    if not Path(TEXT_CSV).is_file():
        raise SystemExit(f"Dataset not found: {TEXT_CSV}")
    if not Path(NLP_MODEL_PATH).is_file() or not Path(NLP_VECTORIZER_PATH).is_file():
        raise SystemExit("Trained NLP model / vectorizer missing — run training notebook first.")

    print(f"[eval] loading {TEXT_CSV}")
    df = pd.read_csv(TEXT_CSV)
    df = prepare_dataframe(df, "statement")
    print(f"[eval] cleaned rows: {len(df):,}")

    y = (df["status"] != NORMAL_LABEL).astype(int).to_numpy()
    X = df["clean_text"].to_numpy()

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = joblib.load(NLP_VECTORIZER_PATH)
    clf = joblib.load(NLP_MODEL_PATH)

    Xv_test = vectorizer.transform(X_test)
    classes = list(clf.classes_)
    idx_neg = classes.index(1) if 1 in classes else 1
    proba = clf.predict_proba(Xv_test)
    p_neg = proba[:, idx_neg]

    if threshold is None:
        preds = (p_neg >= 0.5).astype(int)
        used_thr = 0.5
    else:
        preds = (p_neg >= threshold).astype(int)
        used_thr = float(threshold)

    label_names = ["Positive (Normal)", "Negative (Depression)"]
    acc = float((preds == y_test).mean())
    macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_test, preds, labels=[0, 1], zero_division=0
    )
    per_class = [
        {"class": label_names[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in range(2)
    ]
    try:
        roc = float(roc_auc_score(y_test, p_neg))
    except Exception:  # noqa: BLE001
        roc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_test, p_neg))
    except Exception:  # noqa: BLE001
        pr_auc = float("nan")

    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    out = {
        "threshold": used_thr,
        "n_test": int(len(y_test)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "per_class": per_class,
        "classes": label_names,
        "confusion_matrix": cm.tolist(),
    }

    eval_path = MODELS_DIR / "nlp_eval.json"
    img_path = MODELS_DIR / "nlp_confusion.png"
    eval_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _save_confusion_png(cm, label_names, img_path)

    print()
    print(classification_report(y_test, preds, target_names=label_names, zero_division=0))
    print(
        f"[eval] thr={used_thr:.2f}  acc={acc:.4f}  macroF1={macro_f1:.4f}  "
        f"ROC-AUC={roc:.4f}  PR-AUC={pr_auc:.4f}"
    )
    print(f"[eval] wrote {eval_path}")
    print(f"[eval] wrote {img_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Custom decision threshold for P(negative). Defaults to model's 0.5.",
    )
    args = ap.parse_args()
    evaluate(threshold=args.threshold)


if __name__ == "__main__":
    main()
