import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. 讀取資料
df = pd.read_csv("classification.csv")  # 由第四次作業產生
X = df[["lon", "lat"]].values
y = df["label"].values

# 2. 特徵標準化
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 3. GDA 理論公式
def fit_gda(X, y):
    """
    估計的五個參數：ϕ, μ0, μ1, Σ
    """
    m = X.shape[0]
    phi = np.mean(y)  # P(y=1)
    mu0 = np.mean(X[y == 0], axis=0)
    mu1 = np.mean(X[y == 1], axis=0)
    
    # 變異矩陣 Σ (LDA 形式)
    Sigma = np.zeros((X.shape[1], X.shape[1]))
    for i in range(m):
        xi = X[i].reshape(-1, 1)
        mu = mu1.reshape(-1, 1) if y[i] == 1 else mu0.reshape(-1, 1)
        Sigma += (xi - mu) @ (xi - mu).T
    Sigma /= m

    return phi, mu0, mu1, Sigma

def gaussian_pdf(X, mu, Sigma):
    """
    多變量常態分佈 pdf 
    """
    n = X.shape[1]
    det_Sigma = np.linalg.det(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)
    norm_const = 1.0 / np.sqrt((2 * np.pi) ** n * det_Sigma)
    X_center = X - mu
    exponent = -0.5 * np.sum(X_center @ inv_Sigma * X_center, axis=1)
    return norm_const * np.exp(exponent)

def predict_proba(X, phi, mu0, mu1, Sigma):
    """
    用講義中的貝氏定理計算 P(y=1|x)
    """
    p1 = gaussian_pdf(X, mu1, Sigma)
    p0 = gaussian_pdf(X, mu0, Sigma)
    numerator = phi * p1
    denominator = numerator + (1 - phi) * p0
    return numerator / denominator

# 4. 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.3, random_state=42, stratify=y
)

phi, mu0, mu1, Sigma = fit_gda(X_train, y_train)
proba_test = predict_proba(X_test, phi, mu0, mu1, Sigma)
y_pred = (proba_test >= 0.5).astype(int)

# 5. 結果輸出
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:\n", cm)

# 6. 畫出決策邊界 (remark: boundary is linear)
xx, yy = np.meshgrid(
    np.linspace(X_std[:, 0].min(), X_std[:, 0].max(), 200),
    np.linspace(X_std[:, 1].min(), X_std[:, 1].max(), 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]
prob = predict_proba(grid, phi, mu0, mu1, Sigma)
zz = prob.reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, zz, levels=20, cmap="coolwarm", alpha=0.7)
plt.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
plt.scatter(
    X_std[y == 0, 0], X_std[y == 0, 1],
    color="blue", s=10, label="class 0"
)
plt.scatter(
    X_std[y == 1, 0], X_std[y == 1, 1],
    color="red", s=10, label="class 1"
)
plt.title("Gaussian Discriminant Analysis ")
plt.xlabel("x₁ (standardized lon)")
plt.ylabel("x₂ (standardized lat)")
plt.legend()
plt.colorbar(label="P(y=1|x)")
plt.show()
