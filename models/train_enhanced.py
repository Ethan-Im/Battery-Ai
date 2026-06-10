import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from obelix import OBELiX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# OBELiX 데이터
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

df = pd.DataFrame(rows).dropna(subset=["ionic_conductivity", "formula"])
df["log_conductivity"] = np.log10(df["ionic_conductivity"])

# 아웃라이어 제거 (-15 미만 극단치 제거)
df = df[df["log_conductivity"] > -15]
print(f"아웃라이어 제거 후: {len(df)}개")

ep = ElementProperty.from_preset("magpie")
def safe_featurize(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

df["magpie"] = df["formula"].apply(safe_featurize)
df = df.dropna(subset=["magpie"])

X_magpie = pd.DataFrame(df["magpie"].tolist(), columns=ep.feature_labels())
struct = ["space_group_num", "a", "b", "c", "alpha", "beta", "gamma"]
X_struct = df[struct].fillna(df[struct].median())

# 파생 피처 생성 (격자 체적, 밀도 등)
X_struct["volume"] = X_struct["a"] * X_struct["b"] * X_struct["c"]
X_struct["volume_per_atom"] = X_struct["volume"] / 3  # 임의 원자수
X_struct["a_b_ratio"] = X_struct["a"] / (X_struct["b"] + 1e-5)
X_struct["c_a_ratio"] = X_struct["c"] / (X_struct["a"] + 1e-5)
X_struct["angle_sum"] = X_struct["alpha"] + X_struct["beta"] + X_struct["gamma"]

X_base = pd.concat([X_magpie, X_struct.reset_index(drop=True)], axis=1)
df_base = pd.concat([df[["formula", "log_conductivity"]].reset_index(drop=True), X_base], axis=1)

# CHGNet 피처 합치기
chgnet = pd.read_csv("data/chgnet_embeddings.csv")[["formula", "chgnet_energy", "chgnet_mag"]]
df_merged = pd.merge(df_base, chgnet, on="formula", how="left")
df_merged["chgnet_energy"] = df_merged["chgnet_energy"].fillna(df_merged["chgnet_energy"].median())
df_merged["chgnet_mag"] = df_merged["chgnet_mag"].fillna(df_merged["chgnet_mag"].median())

feature_cols = [c for c in df_merged.columns if c not in ["formula", "log_conductivity"]]
X = df_merged[feature_cols]
y = df_merged["log_conductivity"]

print(f"총 피처: {X.shape[1]}개 | 데이터: {len(X)}개\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost v4
print("XGBoost v4 학습...")
xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_pred)
print(f"XGBoost v4 R²: {xgb_r2:.3f}")

# CatBoost
print("CatBoost 학습...")
cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                        random_state=42, verbose=0)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)
cat_r2 = r2_score(y_test, cat_pred)
cat_mae = mean_absolute_error(y_test, cat_pred)
print(f"CatBoost R²: {cat_r2:.3f}")
print(f"CatBoost MAE: {cat_mae:.3f}")

# 앙상블 (XGB + CatBoost)
ensemble_pred = (xgb_pred + cat_pred) / 2
ensemble_r2 = r2_score(y_test, ensemble_pred)
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
print(f"\n🏆 XGB+CatBoost R²:  {ensemble_r2:.3f}")
print(f"🏆 XGB+CatBoost MAE: {ensemble_mae:.3f}")

import joblib
joblib.dump({"xgb": xgb, "cat": cat}, "models/enhanced_models.pkl")
print("\n모델 저장: models/enhanced_models.pkl")
