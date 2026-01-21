import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.transformers import TransactionDateTransformer
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import joblib


FILEPATH = 'data/Real estate.csv'
MODEL_DIR = Path('models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model_path = MODEL_DIR/"price_prediction_elasticnet.joblib"



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