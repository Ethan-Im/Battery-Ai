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

# 파생 피처 (간단하게)
X_struct_aug = X_struct.copy()
X_struct_aug["volume"] = X_struct["a"] * X_struct["b"] * X_struct["c"]
X_struct_aug["a_b_ratio"] = X_struct["a"] / (X_struct["b"] + 1e-5)

X_base = pd.concat([X_magpie, X_struct_aug.reset_index(drop=True)], axis=1)
df_base = pd.concat([df[["formula", "log_conductivity"]].reset_index(drop=True), X_base], axis=1)

chgnet = pd.read_csv("data/chgnet_embeddings.csv")[["formula", "chgnet_energy", "chgnet_mag"]]
df_merged = pd.merge(df_base, chgnet, on="formula", how="left")
df_merged["chgnet_energy"] = df_merged["chgnet_energy"].fillna(df_merged["chgnet_energy"].median())
df_merged["chgnet_mag"] = df_merged["chgnet_mag"].fillna(df_merged["chgnet_mag"].median())

feature_cols = [c for c in df_merged.columns if c not in ["formula", "log_conductivity"]]
X = df_merged[feature_cols]
y = df_merged["log_conductivity"]

print(f"총 피처: {X.shape[1]}개 | 데이터: {len(X)}개\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost만 테스트
cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                        random_state=42, verbose=0)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_test)
cat_r2 = r2_score(y_test, cat_pred)
cat_mae = mean_absolute_error(y_test, cat_pred)
print(f"CatBoost R²:  {cat_r2:.3f}")
print(f"CatBoost MAE: {cat_mae:.3f}")
