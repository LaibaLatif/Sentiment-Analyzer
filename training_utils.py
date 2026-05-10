"""
Training helpers used by notebooks — keeps `.ipynb` cells short and readable.

Quick start (notebook or shell):
    from notebook_setup import add_project_to_path
    add_project_to_path()
    from training_utils import train_and_save_nlp, train_and_save_cnn

    train_and_save_nlp(run_grid_search=True)        # ~minutes (full CSV)
    train_and_save_cnn(epochs=15, augment=True)     # ~40 min CPU; see per-epoch logs

Improvements vs. the original baseline:
    NLP — word(1-2) + char(3-5) feature union, class-balanced LR, HPT-ready.
    CNN — random flip / affine / color jitter, weight decay, ReduceLROnPlateau,
          best-val checkpoint, optional eval on the held-out FER test folder.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from agents.hpt_agent import HPTAgent

from config import (
    CNN_CLASS_NAMES_JSON,
    CNN_MODEL_PATH,
    CNN_EPOCH_HISTORY_JSON,
    CNN_TRAINING_META_JSON,
    IMG_SIZE,
    IMG_TEST,
    IMG_TRAIN,
    MODELS_DIR,
    NLP_MODEL_PATH,
    NLP_VECTORIZER_PATH,
    NORMAL_LABEL,
    TEXT_CSV,
)
from text_clean import prepare_dataframe


# =============================================================================
# NLP
# =============================================================================
def _build_text_vectorizer() -> FeatureUnion:
    """Word (1-2) + char (3-5) TF-IDF — robust on slang and short text."""
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=60_000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        max_features=40_000,
    )
    return FeatureUnion([("word", word), ("char", char)])


def train_and_save_nlp(
    *,
    run_grid_search: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Full NLP pipeline → save TF-IDF (FeatureUnion) + LogisticRegression.

    Always trains on **the entire** Combined Data.csv. With `run_grid_search=True`
    the HPT Agent picks the best ``C`` / solver / max_iter via cross-validation.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[NLP] Loading {TEXT_CSV}")
    df = pd.read_csv(TEXT_CSV)
    df = prepare_dataframe(df, "statement")
    print(f"[NLP] After cleaning: {len(df):,} rows")
    print(df["status"].value_counts().to_string())

    y = (df["status"] != NORMAL_LABEL).astype(int).to_numpy()
    X = df["clean_text"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("[NLP] Fitting TF-IDF (word 1-2  +  char 3-5)…")
    vectorizer = _build_text_vectorizer()
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)
    print(f"[NLP] Train matrix: {X_train_v.shape}  ·  Test matrix: {X_test_v.shape}")

    hpt_report: dict[str, Any] | None = None
    if run_grid_search:
        print("[NLP] Hyper-parameter tuning (HPTAgent / GridSearchCV)...")
        hpt = HPTAgent(random_state=random_state)
        clf, hpt_report = hpt.tune(X_train_v, y_train, X_test_v, y_test)
        print("[NLP] HPT report:", json.dumps(hpt_report, indent=2))
    else:
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X_train_v, y_train)

    preds = clf.predict(X_test_v)
    test_acc = float((preds == y_test).mean())
    test_f1m = float(f1_score(y_test, preds, average="macro"))
    print(f"[NLP] Hold-out accuracy: {test_acc:.4f}  ·  macro-F1: {test_f1m:.4f}")
    print(
        classification_report(
            y_test, preds, target_names=["Positive (Normal)", "Negative (Depression)"]
        )
    )
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    joblib.dump(vectorizer, NLP_VECTORIZER_PATH)
    joblib.dump(clf, NLP_MODEL_PATH)
    print(f"[NLP] Saved {NLP_VECTORIZER_PATH.name} + {NLP_MODEL_PATH.name}")

    out: dict[str, Any] = {
        "n_samples": int(len(df)),
        "tfidf_shape": tuple(X_train_v.shape),
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1m,
        "model_path": str(NLP_MODEL_PATH),
        "vectorizer_path": str(NLP_VECTORIZER_PATH),
        "grid_search": run_grid_search,
    }
    if hpt_report is not None:
        out["hpt_report"] = hpt_report
    return out


# =============================================================================
# CNN
# =============================================================================
def train_and_save_cnn(
    *,
    epochs: int = 25,
    batch_size: int = 64,
    val_fraction: float = 0.15,
    num_workers: int = 0,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment: bool = True,
    eval_on_test_folder: bool = True,
) -> dict[str, Any]:
    """
    Train ``SmallCNN`` on the **full** ``data/img/train`` directory, with
    augmentation + ReduceLROnPlateau + best-val checkpointing. Also evaluates
    on the held-out ``data/img/test`` folder when ``eval_on_test_folder=True``.

    Notes:
        * On Windows keep ``num_workers=0`` (DataLoader spawn-safety).
        * Default ``epochs=15`` is a good CPU run; use ``10`` for a faster smoke test.
          Raise to 25--35 for extra accuracy (diminishing returns on CPU).
    """
    import torch
    import torch.nn as nn
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms

    from agents.cnn_agent import SmallCNN

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNN] Device: {device}  ·  augment: {augment}  ·  epochs: {epochs}")

    if augment:
        train_tfm = transforms.Compose(
            [
                transforms.Resize(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.05, 0.05)),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
            ]
        )
    else:
        train_tfm = transforms.Compose(
            [transforms.Resize(IMG_SIZE), transforms.ToTensor()]
        )
    eval_tfm = transforms.Compose(
        [transforms.Resize(IMG_SIZE), transforms.ToTensor()]
    )

    full = datasets.ImageFolder(IMG_TRAIN, transform=train_tfm)
    class_names = full.classes
    print(f"[CNN] Train folder: {len(full):,} images  ·  classes: {class_names}")

    n_val = max(1, int(val_fraction * len(full)))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=gen)
    # Use eval transforms on the validation slice (no augmentation)
    val_set.dataset = datasets.ImageFolder(IMG_TRAIN, transform=eval_tfm)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallCNN(len(class_names)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    best_state: dict | None = None
    epoch_history: list[dict[str, Any]] = []

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    print(
        "[CNN] Each epoch: batch progress (within train), then train/val accuracy as %.\n"
        "      Overall training progress = completed epochs / total.\n"
    )
    sys.stdout.flush()

    for epoch in range(epochs):
        overall_pct = 100.0 * epoch / max(1, epochs)
        model.train()
        correct = total = 0
        running = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)
            with torch.inference_mode():
                preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            batch_pct = 100.0 * batch_idx / max(1, n_train_batches)
            # Refresh same line — shows how far through this epoch's training batches
            if batch_idx == 1 or batch_idx == n_train_batches or batch_idx % max(
                1, n_train_batches // 20
            ) == 0:
                print(
                    f"\r  epoch {epoch + 1:02d}/{epochs} "
                    f"(overall {overall_pct:.1f}% done) | "
                    f"train batches {batch_pct:5.1f}% ",
                    end="",
                    flush=True,
                )
        train_acc = correct / max(1, total)
        train_loss = running / max(1, total)
        print()  # newline after \r batch line

        model.eval()
        v_correct = v_total = 0
        with torch.inference_mode():
            for vb_idx, (x, y) in enumerate(val_loader, start=1):
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += y.size(0)
                val_batch_pct = 100.0 * vb_idx / max(1, n_val_batches)
                if vb_idx == 1 or vb_idx == n_val_batches or vb_idx % max(
                    1, n_val_batches // 10
                ) == 0:
                    print(
                        f"\r  epoch {epoch + 1:02d}/{epochs} | "
                        f"validation batches {val_batch_pct:5.1f}% ",
                        end="",
                        flush=True,
                    )
        val_acc = v_correct / max(1, v_total)
        print()

        sched.step(val_acc)
        cur_lr = opt.param_groups[0]["lr"]
        row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_acc": round(val_acc, 6),
            "lr": cur_lr,
        }
        epoch_history.append(row)
        ep_done_pct = 100.0 * (epoch + 1) / max(1, epochs)
        print(
            f"  [done] epoch {epoch + 1:02d}/{epochs} complete "
            f"({ep_done_pct:.1f}% of all epochs) | "
            f"loss={train_loss:.4f} | "
            f"train={train_acc * 100:.2f}% | "
            f"val={val_acc * 100:.2f}% | "
            f"lr={cur_lr:.2e}"
        )
        sys.stdout.flush()

        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out: dict[str, Any] = {
        "classes": class_names,
        "best_val_acc": float(best_val),
        "epochs": epochs,
        "epoch_history": epoch_history,
        "augment": augment,
        "weight_decay": weight_decay,
        "model_path": str(CNN_MODEL_PATH),
        "classes_path": str(CNN_CLASS_NAMES_JSON),
        "device": str(device),
    }

    # Final clean evaluation on the held-out FER test folder (before save — meta is complete)
    test_acc: float | None = None
    t_total = 0
    if eval_on_test_folder and IMG_TEST.is_dir():
        test_ds = datasets.ImageFolder(IMG_TEST, transform=eval_tfm)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        model.eval()
        t_correct = 0
        with torch.inference_mode():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                t_correct += (preds == y).sum().item()
                t_total += y.size(0)
        test_acc = t_correct / max(1, t_total)
        print(
            f"[CNN] FER held-out test: {test_acc * 100:.2f}% "
            f"({test_acc:.4f})  (n={t_total:,})"
        )
        out["test_set_acc"] = float(test_acc)
        out["test_set_size"] = int(t_total)

    training_meta = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "epochs_run": epochs,
        "best_val_acc": float(best_val),
        "test_set_acc": test_acc,
        "test_set_size": t_total if t_total else None,
        "per_epoch": epoch_history,
        "augment": augment,
        "lr": lr,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "val_fraction": val_fraction,
        "n_train_images": n_train,
        "n_val_images": n_val,
    }

    CNN_CLASS_NAMES_JSON.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": len(class_names),
            "training_meta": training_meta,
        },
        CNN_MODEL_PATH,
    )
    CNN_TRAINING_META_JSON.write_text(
        json.dumps(training_meta, indent=2),
        encoding="utf-8",
    )
    CNN_EPOCH_HISTORY_JSON.write_text(
        json.dumps(epoch_history, indent=2),
        encoding="utf-8",
    )
    print(f"[CNN] Saved {CNN_MODEL_PATH.name} + {CNN_CLASS_NAMES_JSON.name}")
    print(f"[CNN] Training summary -> {CNN_TRAINING_META_JSON.name}")
    print(f"[CNN] Per-epoch log -> {CNN_EPOCH_HISTORY_JSON.name}")
    print(f"[CNN] Best validation accuracy: {best_val:.4f}")

    return out
