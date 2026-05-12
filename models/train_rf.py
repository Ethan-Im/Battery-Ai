import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv("data/merged_dataset.csv")
df = df.dropna(subset=["log_conductivity", "band_gap", "formation_energy", "energy_above_hull", "density", "volume", "nelements", "nsites"])

features = ["band_gap", "formation_energy", "energy_above_hull", "density", "volume", "nelements", "nsites"]
X = df[features]
y = df["log_conductivity"]

print(f"학습 데이터: {len(df)}개\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"테스트 MAE: {mae:.3f} (log S/cm)")
print(f"테스트 R²:  {r2:.3f}")

# 피처 중요도
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(f"\n피처 중요도:")
print(importances)

# 예측 vs 실제 그래프
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("실제 log(이온 전도도)")
plt.ylabel("예측 log(이온 전도도)")
plt.title(f"Random Forest (R²={r2:.3f})")
plt.tight_layout()
plt.savefig("models/rf_prediction.png", dpi=150)
print("\n그래프 저장: models/rf_prediction.png")
