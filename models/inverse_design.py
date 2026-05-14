import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from itertools import product
import warnings
warnings.filterwarnings("ignore")

# 학습된 모델 로드
from xgboost import XGBRegressor
model = XGBRegressor()
model.load_model("models/xgb_ionic_v2.json")
ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

# 고체 전해질에 자주 쓰이는 원소들
Li_fixed = True
elements = ["P", "S", "O", "N", "F", "Cl", "Br", "Si", "Ge", "Al", "La", "Zr", "Ti"]

# 랜덤 조성 생성
np.random.seed(42)
candidates = []

print("후보 조성 생성 중...")
for _ in range(5000):
    # 2~4개 원소 랜덤 선택
    n_elem = np.random.randint(2, 5)
    chosen = np.random.choice(elements, n_elem, replace=False).tolist()
    chosen = ["Li"] + chosen

    # 랜덤 비율
    ratios = np.random.randint(1, 8, len(chosen))
    from math import gcd
    from functools import reduce
    g = reduce(gcd, ratios)
    ratios = ratios // g

    formula = "".join(f"{e}{r}" if r > 1 else e for e, r in zip(chosen, ratios))
    candidates.append(formula)

# 피처 생성 & 예측
print("이온 전도도 예측 중...")
results = []
for formula in candidates:
    feats = get_features(formula)
    if feats is not None:
        pred = model.predict([feats])[0]
        results.append({"formula": formula, "predicted_log_conductivity": pred})

df = pd.DataFrame(results)
df = df.sort_values("predicted_log_conductivity", ascending=False)
df["predicted_conductivity"] = 10 ** df["predicted_log_conductivity"]

print(f"\n총 후보: {len(df)}개")
print(f"\n🏆 Top 20 고전도도 후보:")
print(df.head(20)[["formula", "predicted_log_conductivity", "predicted_conductivity"]].to_string(index=False))

df.to_csv("data/inverse_design_candidates.csv", index=False)
print("\n저장 완료: data/inverse_design_candidates.csv")
