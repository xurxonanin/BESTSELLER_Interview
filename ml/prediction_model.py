import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path

FILEPATH = 'data/Real estate.csv'

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR/"price_prediction_elasticnet.joblib"

class TransactionDateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, day: int = 1):
        self.day = day
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        dates = X.iloc[:, 0].astype(float)
        years = np.floor(dates).astype(int)
        months = np.rint((dates - years) * 12).astype(int).clip(1,12)
        days = np.full_like(years, fill_value=int(self.day))

        return np.column_stack([years, months, days])
    
    def get_feature_names_out(self, input_features=None):
        return np.array(['X1 Transaction Year', 'X1 Transaction Month', 'X1 Transaction Day'], dtype=object)

raw_data = pd.read_csv(FILEPATH).dropna()
X_cols = ["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude"]
DATE = ["X1 transaction date"]
NUM = [name for name in X_cols if name not in DATE]
Y_cols = ['Y house price of unit area']
X = raw_data[X_cols]
Y = raw_data[Y_cols]
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.25,
    random_state=250
)
preprocess = ColumnTransformer(
    transformers=[
        ("Transaction Date Formatted", TransactionDateTransformer(day=1), DATE),
        ("Num", StandardScaler(), NUM)
    ],
    verbose_feature_names_out=False
)
pipeline = Pipeline([
    ("Preprocessing", preprocess),
    ("Model", ElasticNet())
])
pipeline.fit(X_train, Y_train)

joblib.dump(pipeline, model_path)
