import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TransactionDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, day: int = 1):
        self.day = day
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        dates = X.iloc[:, 0].astype(float)
        years = np.floor(dates).astype(int)
        months = np.rint((dates - years) * 12).astype(int).clip(1,12)

        return np.column_stack([years, months])
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['X1 Transaction Year', 'X1 Transaction Month'], dtype=object)