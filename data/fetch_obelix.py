from obelix import OBELiX
import pandas as pd

ob = OBELiX()

rows = []
for entry in ob:
    rows.append({
        "formula": entry.get("Reduced Composition"),
        "ionic_conductivity": entry.get("Ionic conductivity (S cm-1)"),
        "space_group": entry.get("Space group"),
        "has_cif": entry.get("cif") is not None,
    })

df = pd.DataFrame(rows)
df = df.dropna(subset=["ionic_conductivity"])

df.to_csv("data/obelix.csv", index=False)
print(f"저장 완료: {len(df)}개")
print(df.head(10))
print(f"\n이온 전도도 범위: {df['ionic_conductivity'].min():.2e} ~ {df['ionic_conductivity'].max():.2e} S/cm")
