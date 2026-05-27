import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from obelix import OBELiX
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 데이터 로드
print("데이터 로드 중...")
ob = OBELiX()
rows = []
for entry in ob:
    rows.append({
        "formula": entry.get("Reduced Composition"),
        "ionic_conductivity": entry.get("Ionic conductivity (S cm-1)"),
        "space_group_num": entry.get("Space group #"),
        "a": entry.get("a"), "b": entry.get("b"), "c": entry.get("c"),
        "alpha": entry.get("alpha"), "beta": entry.get("beta"), "gamma": entry.get("gamma"),
    })

df = pd.DataFrame(rows)
df = df.dropna(subset=["ionic_conductivity", "formula"])
df["log_conductivity"] = np.log10(df["ionic_conductivity"])

ep = ElementProperty.from_preset("magpie")
def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

print("피처 생성 중...")
df["magpie"] = df["formula"].apply(get_features)
df = df.dropna(subset=["magpie"])

X_magpie = pd.DataFrame(df["magpie"].tolist(), columns=ep.feature_labels())
struct_features = ["space_group_num", "a", "b", "c", "alpha", "beta", "gamma"]
X_struct = df[struct_features].fillna(df[struct_features].median())
X = pd.concat([X_magpie, X_struct.reset_index(drop=True)], axis=1)
y = df["log_conductivity"].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 개별 모델 학습
print("\n개별 모델 학습 중...")

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
print(f"RF     R²: {rf_r2:.3f}")

xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost R²: {xgb_r2:.3f}")

lgbm = LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
lgbm_pred = lgbm.predict(X_test)
lgbm_r2 = r2_score(y_test, lgbm_pred)
print(f"LightGBM R²: {lgbm_r2:.3f}")

# 앙상블 (가중 평균)
ensemble_pred = (rf_pred + xgb_pred + lgbm_pred) / 3
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"\n🏆 Ensemble R²:  {ensemble_r2:.3f}")
print(f"🏆 Ensemble MAE: {ensemble_mae:.3f} (log S/cm)")

import joblib
joblib.dump({"rf": rf, "xgb": xgb, "lgbm": lgbm}, "models/ensemble_models.pkl")
print("\n모델 저장: models/ensemble_models.pkl")
