import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from .config import RANDOM_STATE

def rmse_scorer():
    # sklearn expects "higher is better", so we use negative RMSE
    return "neg_root_mean_squared_error"

def make_models():
    models = {
        "DummyMean": DummyRegressor(strategy="mean"),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
    }
    return models

def cross_validate_models(X, y, n_splits: int = 5) -> pd.DataFrame:
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"rmse": rmse_scorer(), "r2": "r2"}

    rows = []
    for name, estimator in make_models().items():
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
        )
        rmse = -scores["test_rmse"]  # negate back to positive RMSE
        r2 = scores["test_r2"]

        rows.append({
            "model": name,
            "cv_folds": n_splits,
            "rmse_mean": float(np.mean(rmse)),
            "rmse_std": float(np.std(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
            "r2_mean": float(np.mean(r2)),
            "r2_std": float(np.std(r2, ddof=1)) if len(r2) > 1 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("rmse_mean")
