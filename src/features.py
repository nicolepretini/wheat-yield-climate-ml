import pandas as pd
from .config import BASE_FEATURES

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a feature matrix from the clean dataframe.
    Keeps base features and adds simple engineered terms.
    """
    X = df[BASE_FEATURES].copy()

    # engineered features (simple, transparent)
    X["tas_sq"] = X["tas_gs_mean"] ** 2
    X["pr_sq"] = X["pr_gs_sum"] ** 2
    X["tas_x_pr"] = X["tas_gs_mean"] * X["pr_gs_sum"]

    return X
