import re, math, csv, os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

XML_PATH = "/content/O-A0038-003.xml"   
CLASS_CSV = "/content/classification.csv"
REGR_CSV  = "/content/regression.csv"

NX, NY = 67, 120          # 經向 67、緯向 120（先經向遞增，再緯向）
DX = DY = 0.03
INVALID = -999.0

# 工具
ns = {"n": "urn:cwa:gov:tw:cwacommon:0.1"}  # CWA XML namespace（若無會自動退回無命名空間方案）

def find_text(root, tag):
    node = root.find(f".//n:{tag}", ns)
    if node is None:
        node = root.find(f".//{tag}")
    return node.text.strip() if node is not None and node.text is not None else None

# 1 解析 XML
tree = ET.parse(XML_PATH)
root = tree.getroot()

blon = float(find_text(root, "BottomLeftLongitude"))
blat = float(find_text(root, "BottomLeftLatitude"))
content_txt = find_text(root, "Content")
if content_txt is None:
    raise ValueError("找不到 <Content> 欄位，請確認 XML 檔案結構。")

# 2 內容 數值向量 
tokens = [t for t in re.split(r"[,\s]+", content_txt) if t != ""]
vals = np.array([float(t) for t in tokens], dtype=np.float32)

expected = NX * NY
if vals.size != expected:
    raise ValueError(f"資料數量不符：期望 {expected}，實得 {vals.size}。請確認 NX/NY 或 XML。")

# 3 重塑為 (NY, NX) 網格
grid = vals.reshape(NY, NX)  # 120×67

# 4 建立經緯度座標
lon = blon + DX * np.arange(NX)            # shape: (67,)
lat = blat + DY * np.arange(NY)            # shape: (120,)
LON, LAT = np.meshgrid(lon, lat)           # shape: (120,67)

xx = LON.ravel()
yy = LAT.ravel()
zz = grid.ravel()

# 5 生成兩個資料集 
# (a) 回歸：僅保留有效值（去除 -999）
mask = zz != INVALID
reg_df = pd.DataFrame({
    "lon": xx[mask],
    "lat": yy[mask],
    "value": zz[mask]
})

# (b) 分類：有效=1，無效=0
cls_df = pd.DataFrame({
    "lon": xx,
    "lat": yy,
    "label": (zz != INVALID).astype(np.int32)
})

# 6 輸出 CSV
reg_df.to_csv(REGR_CSV, index=False)
cls_df.to_csv(CLASS_CSV, index=False)

print(f"✅ 已輸出: {CLASS_CSV}  共 {len(cls_df)} 筆（應為 67×120 = 8040）")
print(f"✅ 已輸出: {REGR_CSV}  共 {len(reg_df)} 筆（已排除 -999）")

# 檢查
print("\nclassification 頭幾列：")
display(cls_df.head())
print("regression 頭幾列：")
display(reg_df.head())
