import pandas as pd
import numpy as np
from obelix import OBELiX
from pathlib import Path

ob = OBELiX()

rows = []
for entry in ob:
    if entry.get("structure") is not None:
        rows.append({
            "formula": entry.get("Reduced Composition"),
            "ionic_conductivity": entry.get("Ionic conductivity (S cm-1)"),
            "structure": entry.get("structure"),
            "id": entry.get("ID"),
        })

df = pd.DataFrame(rows)
df = df.dropna(subset=["ionic_conductivity", "structure"])
df["log_conductivity"] = np.log10(df["ionic_conductivity"])

# CIF 파일 저장
from pymatgen.io.cif import CifWriter
cif_dir = Path("data/cifs")
cif_dir.mkdir(exist_ok=True)

df["cif_path"] = ""
for i, row in df.iterrows():
    path = cif_dir / f"{i}.cif"
    CifWriter(row["structure"]).write_file(str(path))
    df.at[i, "cif_path"] = str(path)

df = df.drop(columns=["structure"])
df.to_csv("data/alignn_dataset.csv", index=False)
print(f"ALIGNN 데이터 준비 완료: {len(df)}개")
print(df[["formula", "log_conductivity"]].head(10))
