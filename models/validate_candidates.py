import pandas as pd
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability
from chgnet.model import CHGNet
from mp_api.client import MPRester
import warnings
warnings.filterwarnings("ignore")

API_KEY = "4c3ufBSs0C5JzgOvwYvGB0dnRCfNz1Z9"
model = CHGNet.load()

df = pd.read_csv("data/final_candidates.csv").head(20)

print("CHGNet 안정성 검증 중...\n")

results = []
with MPRester(API_KEY) as mpr:
    for _, row in df.iterrows():
        formula = row["formula"]
        try:
            # MP에서 유사 구조 검색
            comp = Composition(formula)
            
            # 비슷한 조성 검색
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "energy_above_hull", "formation_energy_per_atom"],
            )
            
            if docs:
                doc = docs[0]
                results.append({
                    "formula": formula,
                    "predicted_log_conductivity": row["log_conductivity"],
                    "predicted_conductivity": row["predicted_conductivity"],
                    "mp_match": doc.formula_pretty,
                    "energy_above_hull": doc.energy_above_hull,
                    "formation_energy": doc.formation_energy_per_atom,
                    "stable": doc.energy_above_hull < 0.1,
                })
                status = "✅ 안정" if doc.energy_above_hull < 0.1 else "⚠️ 불안정"
                print(f"{formula:20s} → {doc.formula_pretty:20s} | hull: {doc.energy_above_hull:.3f} eV | {status}")
            else:
                results.append({
                    "formula": formula,
                    "predicted_log_conductivity": row["log_conductivity"],
                    "predicted_conductivity": row["predicted_conductivity"],
                    "mp_match": None,
                    "energy_above_hull": None,
                    "formation_energy": None,
                    "stable": None,
                })
                print(f"{formula:20s} → MP 매칭 없음 (신규 후보 가능성)")
        except Exception as e:
            print(f"{formula:20s} → 오류: {e}")

df_results = pd.DataFrame(results)
stable = df_results[df_results["stable"] == True]
print(f"\n✅ 안정한 후보: {len(stable)}개")
print(f"❓ MP 미등록 (신규): {df_results['mp_match'].isna().sum()}개")
print(f"\n최종 검증된 후보:")
print(df_results[["formula", "predicted_conductivity", "energy_above_hull", "stable"]].to_string(index=False))

df_results.to_csv("data/validated_candidates.csv", index=False)
print("\n저장 완료: data/validated_candidates.csv")
