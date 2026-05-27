import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from obelix import OBELiX
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# OBELiX 기본 데이터
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
X_base = pd.concat([X_magpie, X_struct.reset_index(drop=True)], axis=1)
df_base = pd.concat([df[["formula", "log_conductivity"]].reset_index(drop=True), X_base], axis=1)

# CHGNet 피처 합치기
chgnet = pd.read_csv("data/chgnet_embeddings.csv")[["formula", "chgnet_energy", "chgnet_mag"]]
df_merged = pd.merge(df_base, chgnet, on="formula", how="left")

# CHGNet 없는 행은 중앙값으로 채우기
df_merged["chgnet_energy"] = df_merged["chgnet_energy"].fillna(df_merged["chgnet_energy"].median())
df_merged["chgnet_mag"] = df_merged["chgnet_mag"].fillna(df_merged["chgnet_mag"].median())

feature_cols = [c for c in df_merged.columns if c not in ["formula", "log_conductivity"]]
X = df_merged[feature_cols]
y = df_merged["log_conductivity"]

print(f"총 피처: {X.shape[1]}개 | 데이터: {len(X)}개")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE: {mae:.3f} (log S/cm)")
print(f"Test R²:  {r2:.3f}")

xgb.save_model("models/xgb_ionic_v4.json")
print("모델 저장: models/xgb_ionic_v4.json")
