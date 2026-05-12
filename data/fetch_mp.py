from mp_api.client import MPRester
import pandas as pd

API_KEY = "4c3ufBSs0C5JzgOvwYvGB0dnRCfNz1Z9"

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["Li"],
        fields=[
            "material_id",
            "formula_pretty",
            "formation_energy_per_atom",
            "energy_above_hull",
            "band_gap",
            "density",
            "volume",
            "nelements",
            "nsites",
        ],
        energy_above_hull=(0, 0.1),
        num_chunks=10,
        chunk_size=1000,
    )

df = pd.DataFrame([{
    "material_id": d.material_id,
    "formula": d.formula_pretty,
    "formation_energy": d.formation_energy_per_atom,
    "energy_above_hull": d.energy_above_hull,
    "band_gap": d.band_gap,
    "density": d.density,
    "volume": d.volume,
    "nelements": d.nelements,
    "nsites": d.nsites,
} for d in docs])

# 후처리 필터: Li 반드시 포함, 2종 이상 원소
df = df[df["formula"].str.contains("Li")]
df = df[df["nelements"] >= 2]

df.to_csv("data/li_materials.csv", index=False)
print(f"저장 완료: {len(df)}개 소재")
print(df.head(10))
print(f"\n원소 수 분포:\n{df['nelements'].value_counts().sort_index()}")
