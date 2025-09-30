import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# 1 載入資料
REGR_CSV = "/content/regression.csv"
df = pd.read_csv(REGR_CSV)

# 2 雙重標準化（特徵 X 與目標 y）
X = df[["lon","lat"]].values.astype(np.float32)
y = df["value"].values.astype(np.float32).reshape(-1,1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_std = x_scaler.fit_transform(X).astype(np.float32)
y_std = y_scaler.fit_transform(y).astype(np.float32)

# 3 資料分割（70%/15%/15%）
X_train, X_tmp, y_train, y_tmp = train_test_split(X_std, y_std, test_size=0.30, random_state=42)
X_val, X_test,  y_val, y_test  = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=512)
test_loader  = DataLoader(test_ds, batch_size=512)

# 4 建立模型（2→16→16→1）
class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,16), nn.ReLU(),
            nn.Linear(16,16), nn.ReLU(),
            nn.Linear(16,1)   # 無激活，直接輸出數值（標準化尺度）
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Regressor().to(device)

# 5 訓練（MSELoss + Adam）
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 30

def train_or_eval(loader, train=True):
    model.train(train)
    total = 0.0
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        if train:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        total += loss.item() * xb.size(0)
    return total/len(loader.dataset)

for ep in range(1, epochs+1):
    tr = train_or_eval(train_loader, True)
    va = train_or_eval(val_loader, False)
    if ep % 5 == 0 or ep == 1:
        print(f"[{ep:02d}] train MSE={tr:.4f} | val MSE={va:.4f}")

# 6 評估（MAE / RMSE；還原到 °C）
model.eval()
with torch.no_grad():
    Xte = torch.from_numpy(X_test).to(device)
    yhat_std = model(Xte).cpu().numpy()
yhat = y_scaler.inverse_transform(yhat_std)  # 還原 °C
ytrue = y_scaler.inverse_transform(y_test)

mae = mean_absolute_error(ytrue, yhat)
rmse = math.sqrt(mean_squared_error(ytrue, yhat))
print(f"Test MAE = {mae:.3f} °C")
print(f"Test RMSE = {rmse:.3f} °C")

# 7 全域預測（還原到原始單位）
X_all_std = x_scaler.transform(df[["lon","lat"]].values.astype(np.float32))
with torch.no_grad():
    y_all_std = model(torch.from_numpy(X_all_std).to(device)).cpu().numpy()
y_all = y_scaler.inverse_transform(y_all_std).ravel()

# 8 視覺化：griddata 空間插值 → 預測曲面；以及誤差分佈
#   (a) 預測曲面（以 lon-lat 規則網格）
lon_min, lon_max = df["lon"].min(), df["lon"].max()
lat_min, lat_max = df["lat"].min(), df["lat"].max()
gx = np.linspace(lon_min, lon_max, 200)
gy = np.linspace(lat_min, lat_max, 200)
GX, GY = np.meshgrid(gx, gy)
Zpred = griddata(points=df[["lon","lat"]].values, values=y_all, xi=(GX, GY), method="linear")

plt.figure(figsize=(7,6))
cf = plt.contourf(GX, GY, Zpred, levels=20)
plt.colorbar(cf, label="Predicted temperature (°C)")
plt.scatter(df["lon"], df["lat"], s=2, c="k", alpha=0.3, label="samples")
plt.title("Regression — predicted surface (griddata)")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend(loc="lower right")
plt.show()

#   (b) 誤差分佈（僅測試集）
err = (yhat - ytrue).ravel()
plt.figure(figsize=(6,4))
plt.hist(err, bins=30)
plt.title(f"Test errors (pred-true) | MAE={mae:.2f}°C RMSE={rmse:.2f}°C")
plt.xlabel("Error (°C)"); plt.ylabel("Count")
plt.show()
