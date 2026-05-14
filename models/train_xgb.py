import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

ob = pd.read_csv("data/obelix.csv")
ob = ob.dropna(subset=["ionic_conductivity", "formula"])
ob["log_conductivity"] = np.log10(ob["ionic_conductivity"])

# Magpie 피처
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

print(f"데이터: {len(X)}개 | 피처: {X.shape[1]}개\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0,
)
xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MAE: {mae:.3f} (log S/cm)")
print(f"Test R²:  {r2:.3f}")

# 피처 중요도 Top 10
importances = pd.Series(xgb.feature_importances_, index=ep.feature_labels())
print(f"\nTop 10 Features:")
print(importances.nlargest(10))

xgb.save_model("models/xgb_ionic.json")
print("\n모델 저장: models/xgb_ionic.json")
