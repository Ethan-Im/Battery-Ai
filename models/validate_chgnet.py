import pandas as pd
import numpy as np
from pymatgen.core import Composition
from pymatgen.core.periodic_table import Element
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/validated_candidates.csv")

# 조성 기반 안정성 휴리스틱
# 실제 알려진 안정 고체 전해질 패밀리와 비교
known_families = {
    "argyrodite": ["Li", "P", "S"],        # Li6PS5X
    "nasicon": ["Li", "P", "O"],            # LAGP, LATP
    "garnet": ["Li", "La", "Zr", "O"],     # LLZO
    "lgps": ["Li", "P", "S", "Ge"],        # Li10GeP2S12
    "halide": ["Li", "Cl"],                 # Li3YCl6
}

results = []
for _, row in df.iterrows():
    formula = row["formula"]
    try:
        comp = Composition(formula)
        elem_set = set([str(e) for e in comp.elements])
        
        # 패밀리 매칭
        matched = []
        for family, elems in known_families.items():
            if all(e in elem_set for e in elems):
                matched.append(family)
        
        # Li 비율
        li_fraction = comp.get_atomic_fraction(Element("Li"))
        
        # 안정성 스코어 (휴리스틱)
        score = 0
        if matched:
            score += 2
        if 0.3 < li_fraction < 0.7:
            score += 1
        if len(elem_set) <= 4:
            score += 1
        
        results.append({
            "formula": formula,
            "predicted_conductivity": row["predicted_conductivity"],
            "matched_families": ", ".join(matched) if matched else "unknown",
            "li_fraction": round(li_fraction, 3),
            "stability_score": score,
            "recommendation": "⭐ 유망" if score >= 3 else "🔍 검토 필요" if score >= 2 else "❌ 낮음"
        })
    except Exception as e:
        print(f"오류: {formula} — {e}")

df_out = pd.DataFrame(results).sort_values("stability_score", ascending=False)
print("🔬 후보 안정성 평가 결과:\n")
print(df_out[["formula", "predicted_conductivity", "matched_families", "stability_score", "recommendation"]].to_string(index=False))

df_out.to_csv("data/validated_candidates_v2.csv", index=False)
print("\n저장 완료: data/validated_candidates_v2.csv")
