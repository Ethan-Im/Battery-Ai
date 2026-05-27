import pandas as pd
import numpy as np
from pymatgen.core import Structure
from chgnet.model import CHGNet
import torch
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/alignn_dataset.csv")
df = df.dropna(subset=["log_conductivity", "cif_path"])
print(f"구조 데이터: {len(df)}개")

model = CHGNet.load()

# CHGNet으로 에너지 + 구조 피처 추출
print("CHGNet 피처 추출 중...")

results = []
valid_idx = []

for i, row in df.iterrows():
    try:
        struct = Structure.from_file(row["cif_path"])
        pred = model.predict_structure(struct)
        
        results.append({
            "formula": row["formula"],
            "log_conductivity": row["log_conductivity"],
            "chgnet_energy": float(pred["e"]),
            "chgnet_mag": float(np.mean(np.abs(pred["m"]))) if pred["m"] is not None else 0.0,
        })
        valid_idx.append(i)
    except Exception as e:
        pass

print(f"추출 완료: {len(results)}개")

df_out = pd.DataFrame(results)
df_out.to_csv("data/chgnet_embeddings.csv", index=False)
print(f"저장 완료: data/chgnet_embeddings.csv")
print(df_out.head())
