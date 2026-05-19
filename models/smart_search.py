import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from math import gcd
from functools import reduce
import warnings
warnings.filterwarnings("ignore")

xgb = XGBRegressor()
xgb.load_model("models/xgb_ionic_v2.json")
ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

def predict(formula):
    feats = get_features(formula)
    if feats is None:
        return None
    return xgb.predict([feats])[0]

def random_formula(elements, seed=None):
    if seed:
        np.random.seed(seed)
    n_elem = np.random.randint(2, 5)
    chosen = ["Li"] + np.random.choice(elements, n_elem, replace=False).tolist()
    ratios = np.random.randint(1, 8, len(chosen))
    g = reduce(gcd, ratios)
    ratios = ratios // g
    return "".join(f"{e}{r}" if r > 1 else e for e, r in zip(chosen, ratios))

def mutate(formula, elements):
    try:
        comp = Composition(formula)
        elem_list = [str(e) for e in comp.elements if str(e) != "Li"]
        if not elem_list:
            return random_formula(elements)
        # 랜덤하게 원소 하나 교체
        new_elems = elem_list.copy()
        idx = np.random.randint(len(new_elems))
        new_elems[idx] = np.random.choice(elements)
        chosen = ["Li"] + new_elems
        ratios = np.random.randint(1, 8, len(chosen))
        g = reduce(gcd, ratios)
        ratios = ratios // g
        return "".join(f"{e}{r}" if r > 1 else e for e, r in zip(chosen, ratios))
    except:
        return random_formula(elements)

elements = ["P", "S", "O", "N", "F", "Cl", "Br", "Si", "Ge", "Al", "La", "Zr"]

# Phase 1: 랜덤 탐색 1000개
print("Phase 1: 랜덤 탐색 중...")
results = []
for i in range(1000):
    f = random_formula(elements)
    pred = predict(f)
    if pred is not None:
        results.append({"formula": f, "log_conductivity": pred})

df = pd.DataFrame(results).sort_values("log_conductivity", ascending=False)
print(f"완료: {len(df)}개 | 최고: {df.iloc[0]['log_conductivity']:.3f}")

# Phase 2: 상위 20개 기반 돌연변이 탐색
print("\nPhase 2: 돌연변이 탐색 중...")
top20 = df.head(20)["formula"].tolist()

for parent in top20:
    for _ in range(50):
        child = mutate(parent, elements)
        pred = predict(child)
        if pred is not None:
            results.append({"formula": child, "log_conductivity": pred})

df_final = pd.DataFrame(results).drop_duplicates("formula")
df_final = df_final.sort_values("log_conductivity", ascending=False)
df_final["predicted_conductivity"] = 10 ** df_final["log_conductivity"]

print(f"\n🏆 Top 20 후보:")
print(df_final.head(20)[["formula", "log_conductivity", "predicted_conductivity"]].to_string(index=False))

df_final.to_csv("data/smart_search_candidates.csv", index=False)
print("\n저장 완료: data/smart_search_candidates.csv")
