import pandas as pd
import numpy as np

mp = pd.read_csv("data/li_electrolyte_candidates.csv")
ob = pd.read_csv("data/obelix.csv")

# formula 정규화 (공백 제거)
mp["formula_clean"] = mp["formula"].str.replace(" ", "")
ob["formula_clean"] = ob["formula"].str.replace(" ", "")

# 조인
merged = pd.merge(ob, mp, on="formula_clean", how="left")

# 이온 전도도 log 변환 (ML 타겟값)
merged["log_conductivity"] = np.log10(merged["ionic_conductivity"])

print(f"OBELiX 전체: {len(ob)}개")
print(f"MP 매칭 성공: {merged['material_id'].notna().sum()}개")
print(f"\n매칭된 소재:")
matched = merged[merged["material_id"].notna()]
print(matched[["formula_clean", "log_conductivity", "band_gap", "energy_above_hull"]].head(10))

merged.to_csv("data/merged_dataset.csv", index=False)
print(f"\n저장 완료: data/merged_dataset.csv")
