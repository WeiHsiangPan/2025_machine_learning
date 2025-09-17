import math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# %% 隨機種子
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# %% 在 [-1,1] 區間產生 2400 個點
N = 2400
x_all = np.random.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)
y_all = runge(x_all).astype(np.float32)

# 按照 5:1 比例分割訓練集與驗證集 (by gpt)
idx = np.random.permutation(N)
cut = int(N * 5/6)
train_idx, val_idx = idx[:cut], idx[cut:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_val,   y_val   = x_all[val_idx],   y_all[val_idx]

# 轉換為 PyTorch tensor (by gpt)
Xtr = torch.from_numpy(x_train); Ytr = torch.from_numpy(y_train)
Xva = torch.from_numpy(x_val);   Yva = torch.from_numpy(y_val)

# 建立 DataLoader (批次訓練) (by gpt)
batch_size = 64
train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

# %% 建立模型
# 模型結構: 輸入層 -> 隱藏層1 -> 隱藏層2 -> 輸出層 (by Assignment_2_question1)
# 每個隱藏層使用 tanh (或 ReLU) 激活函數 (by gpt)
ACTIVATION = "tanh" 

class MLP(nn.Module):
    def __init__(self, hidden=32, activation="tanh"):
        super().__init__()
        self.lin1 = nn.Linear(1, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.lin3 = nn.Linear(hidden, 1)
        if activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()  # 預設使用 tanh

    def forward(self, x):
        # 前向傳遞：線性 -> 激活 -> 線性 -> 激活 -> 線性輸出
        z1 = self.lin1(x); a1 = self.act(z1)
        z2 = self.lin2(a1); a2 = self.act(z2)
        yhat = self.lin3(a2)     # 輸出層為線性，不加激活
        return yhat

# 使用 GPU (若可用)，否則 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(hidden=32, activation=ACTIVATION).to(device)

# 損失函數 (MSE)
criterion = nn.MSELoss()

# 最佳化器 (Adam)；也可以換成 SGD 觀察差異
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% 訓練模型
EPOCHS = 2000
train_losses, val_losses = [], []

for ep in range(1, EPOCHS+1):
    # ---- 訓練階段 ----
    model.train()
    tr_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device); yb = yb.to(device)
        optimizer.zero_grad()           # 梯度清零
        yhat = model(xb)                # 前向傳遞
        loss = criterion(yhat, yb)      # 計算 MSE
        loss.backward()                 # 反向傳播 (by Assignment_2_question1)
        optimizer.step()                # 更新參數
        tr_loss += loss.item() * xb.size(0)
    tr_loss /= len(train_loader.dataset)

    # 驗證階段
    model.eval()
    va_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device); yb = yb.to(device)
            yhat = model(xb)
            loss = criterion(yhat, yb)
            va_loss += loss.item() * xb.size(0)
    va_loss /= len(val_loader.dataset)

    # 紀錄訓練/驗證損失
    train_losses.append(tr_loss); val_losses.append(va_loss)

    # 每 200 個 epoch 印一次進度
    if ep % 200 == 0 or ep == 1:
        print(f"Epoch {ep:4d} | Train MSE: {tr_loss:.6e} | Val MSE: {va_loss:.6e}")

# %% 在測試點上做預測
model.eval()
x_test = np.linspace(-1, 1, 1000, dtype=np.float32).reshape(-1,1)
with torch.no_grad():
    y_true = runge(x_test)
    y_pred = model(torch.from_numpy(x_test).to(device)).cpu().numpy()

# 計算誤差指標
mse = np.mean((y_pred - y_true)**2)
max_err = np.max(np.abs(y_pred - y_true))
print(f"\nTest MSE = {mse:.6e}")
print(f"Max Abs Error = {max_err:.6e}")

# %% 繪圖: 真實函數 vs NN 預測
plt.figure()
plt.plot(x_test, y_true, label="True Runge f(x)")
plt.plot(x_test, y_pred, label="NN Prediction")
plt.legend()
plt.title(f"Runge Approximation (activation={ACTIVATION})")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True, alpha=0.2)
plt.show()

# 繪圖: Loss 曲線
plt.figure()
plt.plot(train_losses, label="Training MSE")
plt.plot(val_losses, label="Validation MSE")
plt.legend()
plt.title("Training and Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.grid(True, alpha=0.2)
plt.show()
