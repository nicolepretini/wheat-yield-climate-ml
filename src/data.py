import pandas as pd
from .config import DATA_PROCESSED

def load_clean_data(filename: str = "wheat_climate_clean.csv") -> pd.DataFrame:
    path = DATA_PROCESSED / filename
    df = pd.read_csv(path)
    return df
