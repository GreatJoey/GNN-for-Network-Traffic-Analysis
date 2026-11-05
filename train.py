import torch, torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, f1_score
from graphs import train_set, test_set, feat_cols

class EdgeGNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden)
        mlp = lambda: nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINEConv(mlp(), edge_dim=in_dim)
        self.conv2 = GINEConv(mlp(), edge_dim=in_dim)
        self.head = nn.Sequential(nn.Linear(2*hidden + in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, d):
        h = F.relu(self.lin_in(d.x))
        h = F.relu(self.conv1(h, d.edge_index, d.edge_attr))
        h = F.relu(self.conv2(h, d.edge_index, d.edge_attr))
        s, t = d.edge_index
        return self.head(torch.cat([h[s], h[t], d.edge_attr], dim=1))   # one prediction per flow
    
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = EdgeGNN(len(feat_cols)).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

pos = sum(int(g.y.sum()) for g in train_set); tot = sum(len(g.y) for g in train_set)
w = torch.tensor([1.0, (tot - pos) / max(pos, 1)]).to(device)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8)

for epoch in range(1, 21):
    model.train()
    for b in train_loader:
        b = b.to(device)
        loss = F.cross_entropy(model(b), b.y, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            ps.append(F.softmax(model(b), 1)[:, 1].cpu()); ys.append(b.y.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    print(f"epoch {epoch} | F1 {f1_score(y, p > 0.5):.4f} | PR-AUC {average_precision_score(y, p):.4f}")