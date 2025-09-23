import math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 隨機種子
seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# 函數與其導數
def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)

def runge_prime(x):
    return -50.0 * x / (1.0 + 25.0 * x**2)**2

# %% 在 [-1,1] 區間產生 2400 個點
N = 2400
x_all = np.random.uniform(-1.0, 1.0, size=(N, 1)).astype(np.float32)
y_all = runge(x_all).astype(np.float32)

# 按照 5:1 比例分割訓練集與驗證集（沿用你的作法）
idx = np.random.permutation(N)
cut = int(N * 5/6)
train_idx, val_idx = idx[:cut], idx[cut:]
x_train, y_train = x_all[train_idx], y_all[train_idx]
x_val,   y_val   = x_all[val_idx],   y_all[val_idx]

# 轉換為 PyTorch tensor
Xtr = torch.from_numpy(x_train); Ytr = torch.from_numpy(y_train)
Xva = torch.from_numpy(x_val);   Yva = torch.from_numpy(y_val)

# 建立 DataLoader（批次訓練）
batch_size = 64
train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=batch_size, shuffle=False)

# %% 建立模型
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
        # （小加分）Xavier 初始化：更穩定
        for m in [self.lin1, self.lin2, self.lin3]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        # 前向傳遞：線性 -> 激活 -> 線性 -> 激活 -> 線性輸出
        z1 = self.lin1(x); a1 = self.act(z1)
        z2 = self.lin2(a1); a2 = self.act(z2)
        yhat = self.lin3(a2)     # 輸出層為線性，不加激活
        return yhat

# 使用 GPU (若可用)，否則 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(hidden=32, activation=ACTIVATION).to(device)

# 損失函數（MSE）
criterion = nn.MSELoss()

# 最佳化器（Adam）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 混合損失的權重
lambda_f  = 1.0
lambda_df = 0.1    # 可在 0.05~0.2 之間試

# %% 訓練模型
EPOCHS = 2000
train_losses, val_losses = [], []              # 總損失
train_f_losses, train_df_losses = [], []       # 分別記錄 f 與 f' 的損失
val_f_losses,   val_df_losses   = [], []

for ep in range(1, EPOCHS+1):
    # ---- 訓練階段 ----
    model.train()
    tr_tot = tr_f = tr_df = 0.0
    for xb, yb in train_loader:
        # 關鍵：要對輸入 x 求導 → 讓 xb 成為 leaf 且 requires_grad=True
        xb = xb.to(device).detach().requires_grad_(True)
        yb = yb.to(device)

        optimizer.zero_grad()                   # 梯度清零
        yhat = model(xb)                        # f̂(x)
        f_loss = criterion(yhat, yb)            # 函數誤差

        # 用 autograd 算 f̂'(x)；保留計算圖讓 total.backward() 能回傳到權重
        dy_dx = torch.autograd.grad(
            outputs=yhat,
            inputs=xb,
            grad_outputs=torch.ones_like(yhat),
            create_graph=True,
            retain_graph=True
        )[0]
        df_loss = criterion(dy_dx, runge_prime(xb))   # 導數誤差

        total = lambda_f * f_loss + lambda_df * df_loss
        total.backward()                       # 反向傳播
        optimizer.step()                       # 更新參數

        bs = xb.size(0)
        tr_tot += total.item() * bs
        tr_f   += f_loss.item() * bs
        tr_df  += df_loss.item() * bs
        xb.requires_grad_(False)

    ntr = len(train_loader.dataset)
    tr_tot /= ntr; tr_f /= ntr; tr_df /= ntr
    train_losses.append(tr_tot); train_f_losses.append(tr_f); train_df_losses.append(tr_df)

    # ---- 驗證階段（要算導數，不能用 no_grad；改用 enable_grad）----
    model.eval()
    va_tot = va_f = va_df = 0.0
    with torch.enable_grad():
        for xb, yb in val_loader:
            xb = xb.to(device).detach().requires_grad_(True)
            yb = yb.to(device)

            yhat  = model(xb)
            f_loss = criterion(yhat, yb)

            dy_dx  = torch.autograd.grad(
                outputs=yhat,
                inputs=xb,
                grad_outputs=torch.ones_like(yhat),
                create_graph=False,
                retain_graph=False
            )[0]
            df_loss = criterion(dy_dx, runge_prime(xb))

            bs = xb.size(0)
            va_f  += f_loss.item() * bs
            va_df += df_loss.item() * bs
            xb.requires_grad_(False)

    nva = len(val_loader.dataset)
    va_f /= nva; va_df /= nva
    va_tot = lambda_f * va_f + lambda_df * va_df
    val_losses.append(va_tot); val_f_losses.append(va_f); val_df_losses.append(va_df)

    # 每 200 個 epoch 印一次進度
    if ep % 200 == 0 or ep == 1:
        print(f"Epoch {ep:4d} | Train total {tr_tot:.3e} (f {tr_f:.3e}, f' {tr_df:.3e}) | "
              f"Val total {va_tot:.3e} (f {va_f:.3e}, f' {va_df:.3e})")

# %% 在測試點上做預測與導數
model.eval()
x_test = np.linspace(-1, 1, 1000, dtype=np.float32).reshape(-1,1)
xt = torch.from_numpy(x_test).to(device).detach().requires_grad_(True)

with torch.enable_grad():
    y_pred_t = model(xt)                         # f̂(x)
    dy_dx_t  = torch.autograd.grad(y_pred_t.sum(), xt)[0]  # f̂'(x)
xt.requires_grad_(False)

y_true  = runge(x_test)
y_pred  = y_pred_t.detach().cpu().numpy()
fp_true = runge_prime(x_test)
fp_pred = dy_dx_t.detach().cpu().numpy()

# 計算誤差指標（函數 & 導數）
mse_f  = float(np.mean((y_pred  - y_true)**2))
mse_df = float(np.mean((fp_pred - fp_true)**2))
max_f  = float(np.max(np.abs(y_pred  - y_true)))
max_df = float(np.max(np.abs(fp_pred - fp_true)))
print(f"\nTest MSE f  = {mse_f:.6e} | Max |f̂-f|  = {max_f:.6e}")
print(f"Test MSE f' = {mse_df:.6e} | Max |f̂'-f'| = {max_df:.6e}")

# %% 繪圖: 真實函數 vs NN 預測
plt.figure()
plt.plot(x_test, y_true, label="True Runge f(x)")
plt.plot(x_test, y_pred, label="NN Prediction")
plt.legend()
plt.title(f"Runge Approximation (activation={ACTIVATION})")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True, alpha=0.2)
plt.show()

# 繪圖: 真實導數 vs NN 導數
plt.figure()
plt.plot(x_test, fp_true, label="True f'(x)")
plt.plot(x_test, fp_pred, label="NN f̂'(x)")
plt.legend()
plt.title("Derivative Approximation")
plt.xlabel("x"); plt.ylabel("f'(x)")
plt.grid(True, alpha=0.2)
plt.show()

# 繪圖: Loss 曲線（總損失 + 分量）
plt.figure()
plt.plot(train_losses, label="Training total")
plt.plot(val_losses,   label="Validation total")
plt.plot(train_f_losses, "--", label="Train f")
plt.plot(val_f_losses,   "--", label="Val f")
plt.plot(train_df_losses, ":", label="Train f'")
plt.plot(val_df_losses,   ":", label="Val f'")
plt.legend()
plt.title("Training and Validation Loss (total / f / f')")
plt.xlabel("Epoch"); plt.ylabel("MSE")
plt.grid(True, alpha=0.2)
plt.show()
