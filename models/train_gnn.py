import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# 구조 → 그래프 변환
def structure_to_graph(cif_path, target):
    struct = Structure.from_file(cif_path)
    
    # 원자 피처 (원자번호, 전기음성도 등)
    atom_features = []
    for site in struct:
        z = site.specie.Z
        en = site.specie.X if site.specie.X else 0
        r = site.specie.atomic_radius or 0
        atom_features.append([z, en, float(r)])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 엣지 (cutoff 반경 내 이웃)
    cutoff = 5.0
    edge_src, edge_dst, edge_attr = [], [], []
    for i, site_i in enumerate(struct):
        neighbors = struct.get_neighbors(site_i, cutoff)
        for neighbor in neighbors:
            j = neighbor.index
            dist = neighbor.nn_distance
            edge_src.append(i)
            edge_dst.append(j)
            edge_attr.append([dist])
    
    if len(edge_src) == 0:
        return None
    
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

# 데이터 로드
df = pd.read_csv("data/alignn_dataset.csv")
df = df.dropna(subset=["log_conductivity"])
print(f"전체 데이터: {len(df)}개")
print("그래프 변환 중...")

graphs = []
for _, row in df.iterrows():
    try:
        g = structure_to_graph(row["cif_path"], row["log_conductivity"])
        if g is not None:
            graphs.append(g)
    except:
        pass

print(f"유효 그래프: {len(graphs)}개")

# Train/Test split
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=16)

# GNN 모델 정의
class CrystalGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, 64)
        self.conv1 = CGConv(64, dim=1)
        self.conv2 = CGConv(64, dim=1)
        self.conv3 = CGConv(64, dim=1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, data):
        x = self.embed(data.x).relu()
        x = self.conv1(x, data.edge_index, data.edge_attr).relu()
        x = self.conv2(x, data.edge_index, data.edge_attr).relu()
        x = self.conv3(x, data.edge_index, data.edge_attr).relu()
        x = global_mean_pool(x, data.batch)
        return self.fc(x).squeeze()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"디바이스: {device}")

model = CrystalGNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# 학습
print("\n학습 시작...")
for epoch in range(1, 101):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = nn.MSELoss()(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f}")

# 평가
model.eval()
preds, trues = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = model(batch)
        preds.extend(pred.cpu().numpy())
        trues.extend(batch.y.cpu().numpy())

mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)
print(f"\nTest MAE: {mae:.3f} (log S/cm)")
print(f"Test R²:  {r2:.3f}")

torch.save(model.state_dict(), "models/gnn_ionic.pt")
print("모델 저장: models/gnn_ionic.pt")
