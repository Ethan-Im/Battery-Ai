import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from obelix import OBELiX
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# OBELiX 전체 로드 (구조 피처 포함)
print("OBELiX 로드 중...")
ob = OBELiX()

rows = []
for entry in ob:
    rows.append({
        "formula": entry.get("Reduced Composition"),
        "ionic_conductivity": entry.get("Ionic conductivity (S cm-1)"),
        "space_group_num": entry.get("Space group #"),
        "a": entry.get("a"),
        "b": entry.get("b"),
        "c": entry.get("c"),
        "alpha": entry.get("alpha"),
        "beta": entry.get("beta"),
        "gamma": entry.get("gamma"),
    })

df = pd.DataFrame(rows)
df = df.dropna(subset=["ionic_conductivity", "formula"])
df["log_conductivity"] = np.log10(df["ionic_conductivity"])
print(f"전체 데이터: {len(df)}개")

# Magpie 피처
ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

print("Magpie 피처 생성 중...")
df["magpie"] = df["formula"].apply(get_features)
df = df.dropna(subset=["magpie"])

X_magpie = pd.DataFrame(df["magpie"].tolist(), columns=ep.feature_labels())

# 구조 피처 추가
struct_features = ["space_group_num", "a", "b", "c", "alpha", "beta", "gamma"]
X_struct = df[struct_features].fillna(df[struct_features].median())

# 합치기
X = pd.concat([X_magpie, X_struct.reset_index(drop=True)], axis=1)
y = df["log_conductivity"].reset_index(drop=True)

print(f"총 피처 수: {X.shape[1]}개 (Magpie {X_magpie.shape[1]} + 구조 {len(struct_features)})")
print(f"유효 데이터: {len(X)}개\n")

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
    n_iter=30, cv=5,
    scoring="r2",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("하이퍼파라미터 탐색 중...")
search.fit(X_train, y_train)

best = search.best_estimator_
y_pred = best.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MAE: {mae:.3f} (log S/cm)")
print(f"Test R²:  {r2:.3f}")

# 피처 중요도 Top 10
importances = pd.Series(best.feature_importances_, index=X.columns)
print(f"\nTop 10 Features:")
print(importances.nlargest(10))

best.save_model("models/xgb_ionic_v3.json")
print("\n모델 저장: models/xgb_ionic_v3.json")
