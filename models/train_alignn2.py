import pandas as pd
import numpy as np
from pathlib import Path
from alignn.pretrained import get_figshare_model
from alignn.graphs import Graph
from jarvis.core.atoms import Atoms
from pymatgen.io.jarvis import JarvisAtomsAdaptor
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/alignn_dataset.csv")
df = df.dropna(subset=["log_conductivity"])
print(f"학습 데이터: {len(df)}개")

# 구조 → 그래프 변환
from pymatgen.core import Structure

def structure_to_graph(cif_path):
    try:
        struct = Structure.from_file(cif_path)
        atoms = JarvisAtomsAdaptor.get_atoms(struct)
        g, lg = Graph.atom_dgl_multigraph(atoms)
        return g, lg
    except:
        return None, None

print("그래프 변환 중... (시간이 걸려요)")
graphs = []
valid_idx = []
for i, row in df.iterrows():
    g, lg = structure_to_graph(row["cif_path"])
    if g is not None:
        graphs.append((g, lg))
        valid_idx.append(i)

df_valid = df.loc[valid_idx].reset_index(drop=True)
print(f"유효 그래프: {len(graphs)}개")

# 사전학습 ALIGNN 로드
print("사전학습 ALIGNN 모델 로드 중...")
model = get_figshare_model("jv_formation_energy_peratom_alignn")
print("모델 로드 완료!")
