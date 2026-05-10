"""Evaluate the trained CNN on the held-out FER test folder.

Writes ``models/cnn_eval.json`` (per-class precision / recall / F1, support, accuracy,
macro / weighted F1) and ``models/cnn_confusion.png`` (confusion matrix image).

Run from project root:
    python -m scripts.eval_cnn               # default: no TTA
    python -m scripts.eval_cnn --tta         # use horizontal-flip TTA at inference
    python -m scripts.eval_cnn --batch 128

These artefacts are read by the Streamlit sidebar / Report tab when present.
The trained weights are NOT modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.cnn_agent import SmallCNN  # noqa: E402
from config import (  # noqa: E402
    CNN_CLASS_NAMES_JSON,
    CNN_MODEL_PATH,
    IMG_SIZE,
    IMG_TEST,
    MODELS_DIR,
)


def _load_model(num_classes: int) -> tuple[SmallCNN, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        payload = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(CNN_MODEL_PATH, map_location=device)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if isinstance(payload, dict):
        num_classes = int(payload.get("num_classes", num_classes))
    model = SmallCNN(num_classes).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _save_confusion_png(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    """Save a labeled confusion-matrix PNG (no matplotlib styles, headless-safe)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=120)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title("CNN — confusion matrix (FER held-out test)", fontsize=11)

    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if cm[i, j] > thresh else "#1e293b",
            )
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def evaluate(*, tta: bool, batch_size: int) -> dict:
    if not Path(IMG_TEST).is_dir():
        raise SystemExit(f"Test folder not found: {IMG_TEST}")
    if not Path(CNN_MODEL_PATH).is_file():
        raise SystemExit(f"Trained weights not found: {CNN_MODEL_PATH}")

    class_names = json.loads(Path(CNN_CLASS_NAMES_JSON).read_text(encoding="utf-8"))
    eval_tfm = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.ToTensor()]
    )
    ds = datasets.ImageFolder(IMG_TEST, transform=eval_tfm)
    if list(ds.classes) != list(class_names):
        print(
            f"[warn] dataset classes {ds.classes} differ from {class_names}; "
            "using dataset class order."
        )
        class_names = ds.classes

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model, device = _load_model(num_classes=len(class_names))

    y_true: list[int] = []
    y_pred: list[int] = []
    print(f"[eval] device={device} · n_test={len(ds):,} · tta={tta}")
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if tta:
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())

    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    acc = float((y_pred_a == y_true_a).mean())

    p, r, f, s = precision_recall_fscore_support(
        y_true_a, y_pred_a, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = [
        {
            "class": class_names[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(class_names))
    ]

    macro_f1 = float(f1_score(y_true_a, y_pred_a, average="macro", zero_division=0))
    weighted_f1 = float(
        f1_score(y_true_a, y_pred_a, average="weighted", zero_division=0)
    )
    cm = confusion_matrix(y_true_a, y_pred_a, labels=list(range(len(class_names))))

    out = {
        "tta": tta,
        "n_test": int(len(ds)),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "classes": list(class_names),
        "confusion_matrix": cm.tolist(),
    }

    eval_path = MODELS_DIR / "cnn_eval.json"
    img_path = MODELS_DIR / "cnn_confusion.png"
    eval_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    _save_confusion_png(cm, list(class_names), img_path)

    print()
    print(classification_report(y_true_a, y_pred_a, target_names=class_names, zero_division=0))
    print(f"[eval] accuracy = {acc:.4f}  ·  macro-F1 = {macro_f1:.4f}  ·  weighted-F1 = {weighted_f1:.4f}")
    print(f"[eval] wrote {eval_path}")
    print(f"[eval] wrote {img_path}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tta", action="store_true", help="Average over horizontal flip at inference")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    evaluate(tta=args.tta, batch_size=args.batch)


if __name__ == "__main__":
    main()
