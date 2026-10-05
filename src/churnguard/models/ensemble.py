"""Calibrated Ensemble Classifier for Telco Churn Prediction (v2.0 Architecture).

Blends calibrated probabilities from:
1. Logistic Regression (M2 feature set: drops gender, SeniorCitizen, scaled numeric)
2. Optuna-tuned LightGBM Classifier
3. XGBoost Classifier

Optimizes ensemble soft-voting weights via Out-Of-Fold Brier score minimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from churnguard.config import SEED
from churnguard.features.build import build_full_pipeline
from churnguard.models.calibrate import load_champion_params


class CalibratedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Calibrated Soft-Voting Stacking Ensemble of Logistic Regression, LightGBM, and XGBoost.

    Combines:
    - Estimator 1: Logistic Regression + Sigmoid Calibration (smooth log-odds baseline)
    - Estimator 2: LightGBM + Isotonic Calibration (non-linear interactions & high recall)
    - Estimator 3: XGBoost + Sigmoid Calibration (robust tree regularization)
    """

    def __init__(
        self,
        weights: list[float] | np.ndarray | None = None,
        drop_cols: list[str] | None = None,
        include_interactions: bool = True,
        cv_splits: int = 5,
        random_state: int = SEED,
    ) -> None:
        self.weights = weights
        self.drop_cols = drop_cols if drop_cols is not None else ["gender", "SeniorCitizen"]
        self.include_interactions = include_interactions
        self.cv_splits = cv_splits
        self.random_state = random_state

        self.models_: list[Any] = []
        self.weights_: np.ndarray | None = None
        self.classes_: np.ndarray = np.array([0, 1])

    def _build_base_pipelines(self) -> list[tuple[str, Any]]:
        """Construct the 3 base pipelines."""
        # 1. Logistic Regression Pipeline
        lr_model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state,
        )
        pipe_lr = build_full_pipeline(
            model=lr_model,
            include_engineered=True,
            include_interactions=self.include_interactions,
            scale_numeric=True,
            drop_cols=self.drop_cols,
        )

        # 2. LightGBM Pipeline
        lgb_params = load_champion_params()
        pipe_lgb = build_full_pipeline(
            model=LGBMClassifier(**lgb_params),
            include_engineered=True,
            include_interactions=self.include_interactions,
            scale_numeric=False,
            drop_cols=self.drop_cols,
        )

        # 3. XGBoost Pipeline
        pos_weight = (1.0 - 0.2653) / 0.2653  # ~2.77
        xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
        )
        pipe_xgb = build_full_pipeline(
            model=xgb_model,
            include_engineered=True,
            include_interactions=self.include_interactions,
            scale_numeric=False,
            drop_cols=self.drop_cols,
        )

        return [
            ("lr_sigmoid", CalibratedClassifierCV(pipe_lr, method="sigmoid", cv=self.cv_splits)),
            (
                "lgb_isotonic",
                CalibratedClassifierCV(pipe_lgb, method="isotonic", cv=self.cv_splits),
            ),
            ("xgb_sigmoid", CalibratedClassifierCV(pipe_xgb, method="sigmoid", cv=self.cv_splits)),
        ]

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> CalibratedEnsembleClassifier:
        """Fit base calibrated estimators and optimize ensemble weights on OOF predictions."""
        X_df = X.copy()
        y_arr = np.asarray(y).astype(int)

        base_estimators = self._build_base_pipelines()
        skf = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=self.random_state
        )

        # Compute out-of-fold predictions to optimize blending weights
        oof_predictions: list[np.ndarray] = []
        for _name, est in base_estimators:
            oof_p = cross_val_predict(est, X_df, y_arr, cv=skf, method="predict_proba")[:, 1]
            oof_predictions.append(oof_p)

        oof_mat = np.column_stack(oof_predictions)  # shape (N, 3)

        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            self.weights_ = w / np.sum(w)
        else:
            # Optimize weights to minimize Brier score loss
            def loss_func(w: np.ndarray) -> float:
                w_norm = w / np.sum(w)
                blended = np.dot(oof_mat, w_norm)
                return float(brier_score_loss(y_arr, blended))

            init_w = np.ones(3) / 3.0
            bounds = [(0.0, 1.0) for _ in range(3)]
            constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            opt_res = minimize(
                loss_func,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if opt_res.success:
                self.weights_ = opt_res.x / np.sum(opt_res.x)
            else:
                self.weights_ = init_w

        # Fit final calibrated estimators on full training data
        self.models_ = []
        for _name, est in base_estimators:
            est.fit(X_df, y_arr)
            self.models_.append(est)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calculate weighted soft-voting calibrated probabilities."""
        if not self.models_ or self.weights_ is None:
            raise RuntimeError("CalibratedEnsembleClassifier must be fitted before predict_proba.")

        prob_list = [model.predict_proba(X)[:, 1] for model in self.models_]
        prob_mat = np.column_stack(prob_list)
        blended_pos = np.dot(prob_mat, self.weights_)
        blended_pos = np.clip(blended_pos, 0.0, 1.0)

        return np.column_stack([1.0 - blended_pos, blended_pos])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels given a decision threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
