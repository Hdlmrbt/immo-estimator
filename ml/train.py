import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 1. Données
df = fetch_california_housing(as_frame=True).frame
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# 2. Même découpage 70/15/15 que dans le notebook
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# 3. Pipeline (mets ici les meilleurs hyperparamètres trouvés)
def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=300, max_leaf_nodes=31, random_state=42)),
    ])

# 4. Réentraînement sur train + validation
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
pipe = build_pipeline()
pipe.fit(X_trainval, y_trainval)

# 5. Évaluation finale, UNE seule fois, sur le test
pred = pipe.predict(X_test)
metrics = {
    "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
    "MAE": float(mean_absolute_error(y_test, pred)),
    "R2": float(r2_score(y_test, pred)),
}
print("Résultats sur le test :", metrics)

# 6. Sauvegarde
joblib.dump(pipe, MODELS_DIR / "pipeline.joblib")
(MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
print("Pipeline sauvegardé.")