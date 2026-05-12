import pandas as pd

df = pd.read_csv("data/li_materials.csv")

print(f"전체 소재: {len(df)}개\n")

# 고체 전해질 후보 필터
electrolyte = df[
    (df["band_gap"] > 3.0) &
    (df["energy_above_hull"] < 0.05) &
    (df["nelements"] >= 3)
].copy()

electrolyte = electrolyte.sort_values("energy_above_hull")

print(f"전해질 후보: {len(electrolyte)}개\n")
print(electrolyte[["material_id", "formula", "band_gap", "energy_above_hull", "density"]].head(20))

electrolyte.to_csv("data/li_electrolyte_candidates.csv", index=False)
print(f"\n후보 저장 완료: data/li_electrolyte_candidates.csv")
