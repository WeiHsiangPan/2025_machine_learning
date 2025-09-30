import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

# 1 載入資料
CLASS_CSV = "/content/classification.csv"
df = pd.read_csv(CLASS_CSV)

# 2 特徵標準化（經緯度）
X = df[["lon","lat"]].values.astype(np.float32)
y = df["label"].values.astype(np.float32)

x_scaler = StandardScaler()
X_std = x_scaler.fit_transform(X).astype(np.float32)

# 3 資料分割（60% / 20% / 20%）
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_std, y, test_size=0.40, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

# 4 建立模型（2→16→32→1）
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,16), nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,1)  # logits
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Classifier().to(device)

# 5 訓練（BCEWithLogitsLoss + Adam）
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 25

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device).unsqueeze(1)
        logits = model(xb)
        loss = criterion(logits, yb)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

for ep in range(1, epochs+1):
    tr_loss = run_epoch(train_loader, True)
    va_loss = run_epoch(val_loader,   False)
    if ep % 5 == 0 or ep == 1:
        print(f"[{ep:02d}] train loss={tr_loss:.4f} | val loss={va_loss:.4f}")

# 6 評估（AUC-ROC 計算；輸出機率 0~1）
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()   # 機率 0~1
        all_probs.append(probs)
        all_labels.append(yb.numpy().ravel())
y_score = np.concatenate(all_probs)
y_true  = np.concatenate(all_labels)
test_auc = roc_auc_score(y_true, y_score)
print(f"\nTest AUC = {test_auc:.4f}")

# 7 顯示分類結果（報表 + 混淆矩陣 + 視覺化）
y_pred = (y_score >= 0.5).astype(int)

print("\nClassification report (test):")
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_true, y_pred, digits=4))

print("Confusion matrix (test) [rows=true, cols=pred]:")
print(confusion_matrix(y_true, y_pred))

# 還原測試集的原始經緯度，做視覺化
test_lonlat = pd.DataFrame(
    x_scaler.inverse_transform(X_test), columns=["lon","lat"]
)
vis_df = test_lonlat.copy()
vis_df["label_true"] = y_test.astype(int)
vis_df["prob"] = y_score
vis_df["pred"] = y_pred

# (a) ROC 曲線
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, lw=2, label=f"AUC = {test_auc:.3f}")
plt.plot([0,1],[0,1],"--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC curve (test)")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# (b) 測試集機率熱度圖（lon–lat）
plt.figure(figsize=(7,6))
sc = plt.scatter(vis_df["lon"], vis_df["lat"], c=vis_df["prob"], s=10, alpha=0.9)
plt.colorbar(sc, label="Predicted probability (Valid=1)")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Test set probability map (no decision boundary)")
plt.show()

# (c) 機率分佈直方圖（依真實標籤分群）
plt.figure(figsize=(6,4))
plt.hist(vis_df[vis_df.label_true==1]["prob"], bins=30, alpha=0.6, label="True=1")
plt.hist(vis_df[vis_df.label_true==0]["prob"], bins=30, alpha=0.6, label="True=0")
plt.xlabel("Predicted probability"); plt.ylabel("Count")
plt.title("Probability distribution by true class (test)")
plt.legend()
plt.show()
