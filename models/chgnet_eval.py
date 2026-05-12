import pandas as pd
from chgnet.model import CHGNet
from pymatgen.core import Structure
from mp_api.client import MPRester
import json

API_KEY = "4c3ufBSs0C5JzgOvwYvGB0dnRCfNz1Z9"

# 후보 로드 (일단 상위 20개만 테스트)
df = pd.read_csv("data/li_electrolyte_candidates.csv").head(20)
model = CHGNet.load()

results = []

with MPRester(API_KEY) as mpr:
    for _, row in df.iterrows():
        try:
            # 구조 가져오기
            structure = mpr.get_structure_by_material_id(row["material_id"])
            # CHGNet 예측
            prediction = model.predict_structure(structure)
            results.append({
                "material_id": row["material_id"],
                "formula": row["formula"],
                "mp_formation_energy": row["formation_energy"],
                "chgnet_energy": prediction["e"],
                "band_gap": row["band_gap"],
                "energy_above_hull": row["energy_above_hull"],
            })
            print(f"✅ {row['formula']}: {prediction['e']:.4f} eV/atom")
        except Exception as e:
            print(f"❌ {row['formula']}: {e}")

results_df = pd.DataFrame(results)
results_df.to_csv("data/chgnet_results.csv", index=False)
print(f"\n완료: {len(results_df)}개 평가")
print(results_df.head())
