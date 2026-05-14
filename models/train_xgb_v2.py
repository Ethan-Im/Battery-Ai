import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

ob = pd.read_csv("data/obelix.csv")
ob = ob.dropna(subset=["ionic_conductivity", "formula"])
ob["log_conductivity"] = np.log10(ob["ionic_conductivity"])

ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

print("Magpie 피처 생성 중...")
ob["features"] = ob["formula"].apply(get_features)
ob = ob.dropna(subset=["features"])

X = pd.DataFrame(ob["features"].tolist(), columns=ep.feature_labels())
y = ob["log_conductivity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 탐색
param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
}

xgb = XGBRegressor(random_state=42, verbosity=0)
search = RandomizedSearchCV(
    xgb, param_grid,
    n_iter=30,
    cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("하이퍼파라미터 탐색 중... (2~3분 소요)")
search.fit(X_train, y_train)

print(f"\n최적 파라미터: {search.best_params_}")

best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE: {mae:.3f} (log S/cm)")
print(f"Test R²:  {r2:.3f}")

best_model.save_model("models/xgb_ionic_v2.json")
print("모델 저장: models/xgb_ionic_v2.json")
