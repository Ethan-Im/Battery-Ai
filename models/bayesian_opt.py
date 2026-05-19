import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
import warnings
warnings.filterwarnings("ignore")

# XGBoost 모델 로드 (블랙박스 함수)
xgb = XGBRegressor()
xgb.load_model("models/xgb_ionic_v2.json")
ep = ElementProperty.from_preset("magpie")

def get_features(formula):
    try:
        return ep.featurize(Composition(formula))
    except:
        return None

def predict_conductivity(formula):
    feats = get_features(formula)
    if feats is None:
        return None
    return xgb.predict([feats])[0]

# 탐색할 원소 공간
elements = ["P", "S", "O", "N", "F", "Cl", "Br", "Si", "Ge", "Al", "La", "Zr"]

def random_formula():
    n_elem = np.random.randint(2, 5)
    chosen = ["Li"] + np.random.choice(elements, n_elem, replace=False).tolist()
    ratios = np.random.randint(1, 8, len(chosen))
    from math import gcd
    from functools import reduce
    g = reduce(gcd, ratios)
    ratios = ratios // g
    return "".join(f"{e}{r}" if r > 1 else e for e, r in zip(chosen, ratios))

# 초기 랜덤 탐색 (50개)
print("초기 랜덤 탐색 중...")
init_results = []
while len(init_results) < 50:
    f = random_formula()
    pred = predict_conductivity(f)
    if pred is not None:
        init_results.append({"formula": f, "log_conductivity": pred})

df_obs = pd.DataFrame(init_results)
print(f"초기 탐색 완료: {len(df_obs)}개")
print(f"초기 최고: {df_obs['log_conductivity'].max():.3f} log S/cm")

# Bayesian 최적화 루프
print("\nBayesian 최적화 시작...")
best_results = []

for iteration in range(10):
    # GP 피처: log_conductivity 값들을 입력으로
    X = torch.tensor(df_obs["log_conductivity"].values, dtype=torch.float64).unsqueeze(-1)
    Y = torch.tensor(df_obs["log_conductivity"].values, dtype=torch.float64).unsqueeze(-1)
    
    # GP 모델 학습
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # 새 후보 100개 랜덤 생성 후 EI로 선택
    candidates = []
    for _ in range(100):
        f = random_formula()
        pred = predict_conductivity(f)
        if pred is not None:
            candidates.append({"formula": f, "log_conductivity": pred})
    
    if not candidates:
        continue
    
    df_cand = pd.DataFrame(candidates)
    
    # EI 계산
    best_y = df_obs["log_conductivity"].max()
    X_cand = torch.tensor(df_cand["log_conductivity"].values, dtype=torch.float64).unsqueeze(-1)
    
    ei = ExpectedImprovement(gp, best_f=torch.tensor(best_y, dtype=torch.float64))
    with torch.no_grad():
        ei_vals = ei(X_cand.unsqueeze(1))
    
    # EI 최고 후보 선택
    best_idx = ei_vals.argmax().item()
    best_candidate = df_cand.iloc[best_idx]
    
    df_obs = pd.concat([df_obs, pd.DataFrame([best_candidate])], ignore_index=True)
    best_results.append(best_candidate)
    
    print(f"Iter {iteration+1:2d} | Best so far: {df_obs['log_conductivity'].max():.3f} | New: {best_candidate['formula']} ({best_candidate['log_conductivity']:.3f})")

# 최종 결과
df_obs = df_obs.sort_values("log_conductivity", ascending=False)
df_obs["predicted_conductivity"] = 10 ** df_obs["log_conductivity"]

print(f"\n🏆 Bayesian 최적화 Top 20:")
print(df_obs.head(20)[["formula", "log_conductivity", "predicted_conductivity"]].to_string(index=False))

df_obs.to_csv("data/bayesian_candidates.csv", index=False)
print("\n저장 완료: data/bayesian_candidates.csv")
