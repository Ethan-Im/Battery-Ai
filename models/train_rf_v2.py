import pandas as pd
import numpy as np
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.base import MultipleFeaturizer
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# OBELiX 전체 사용 (MP 매칭 불필요)
ob = pd.read_csv("data/obelix.csv")
ob = ob.dropna(subset=["ionic_conductivity", "formula"])
ob["log_conductivity"] = np.log10(ob["ionic_conductivity"])

print(f"전체 데이터: {len(ob)}개")

# Magpie 조성 피처 생성
ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        comp = Composition(formula)
        return ep.featurize(comp)
    except:
        return None

print("Magpie 피처 생성 중...")
ob["features"] = ob["formula"].apply(get_features)
ob = ob.dropna(subset=["features"])

X = pd.DataFrame(ob["features"].tolist(), columns=ep.feature_labels())
y = ob["log_conductivity"]

print(f"피처 수: {X.shape[1]}개")
print(f"유효 데이터: {len(X)}개\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"테스트 MAE: {mae:.3f} (log S/cm)")
print(f"테스트 R²:  {r2:.3f}")

# 상위 10개 피처
importances = pd.Series(rf.feature_importances_, index=ep.feature_labels())
print(f"\n중요 피처 Top 10:")
print(importances.nlargest(10))
