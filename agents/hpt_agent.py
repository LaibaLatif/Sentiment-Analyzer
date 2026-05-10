"""HPT Agent — hyperparameter tuning for the NLP (TF-IDF + Logistic Regression) path.

Runs at **training time** (not on each Streamlit click). Acts as a self-optimizing
layer over LogisticRegression hyperparameters via GridSearchCV.
"""
from __future__ import annotations

from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV


class HPTAgent:
    """
    Explores a bounded grid (avoid explosion of combinations) and returns
    JSON-serializable metrics + the best sklearn estimator.
    """

    # Compact grid — fast enough for coursework; expand cautiously.
    DEFAULT_GRID: dict[str, list] = {
        "C": [0.25, 0.5, 1.0, 2.0, 4.0],
        "max_iter": [1000, 2000],
        "solver": ["saga", "lbfgs"],
    }

    def __init__(
        self,
        *,
        cv: int = 3,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state

    def tune(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        y_test: Any,
        param_grid: dict[str, list] | None = None,
        *,
        verbose: int = 1,
    ) -> tuple[LogisticRegression, dict[str, Any]]:
        grid_spec = param_grid or self.DEFAULT_GRID
        base = LogisticRegression(
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        grid = GridSearchCV(
            base,
            grid_spec,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=verbose,
            refit=True,
        )
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        preds = best.predict(X_test)
        out: dict[str, Any] = {
            "best_parameters": dict(grid.best_params_),
            "best_cv_score": float(grid.best_score_),
            "test_accuracy": float(accuracy_score(y_test, preds)),
            "test_f1_macro": float(f1_score(y_test, preds, average="macro")),
        }
        return best, out
