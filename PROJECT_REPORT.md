# Kaggle：Echoes of Silenced Genes — 專案報告

---

## 一、競賽任務說明

**問題**：給定一個基因被 CRISPR 敲低（knockdown），預測另外 **5,127 個基因**的差異表現（Differential Expression, DE）。

**差異表現的定義**：
```
DE[擾動基因, 目標基因] = mean(擾動後細胞的表達) - mean(非目標對照細胞的表達)
```
> 簡單來說，就是「這個基因被敲低後，其他基因的表達量變化了多少」。

**評估指標**：Weighted MAE（加權平均絕對誤差）
- 每個擾動 × 每個基因都有不同的權重
- 分數越低越好
- 零預測器（全部預測 0）：WMAE = 0.1413
- 官方 baseline（預測全體平均 DE）：WMAE = 0.1268

**最大挑戰**：訓練集有 80 個擾動基因，測試集有 120 個，且**完全不重疊**。
模型必須對從未見過的基因做出預測（零樣本泛化）。

---

## 二、資料概覽

| 資料集 | 大小 | 說明 |
|--------|------|------|
| `training_cells.h5ad` | 17,882 cells × 19,226 genes | 原始單細胞 RNA-seq 資料 |
| `training_data_ground_truth_table.csv` | 80 × 10,256 | 80個擾動的 DE 值 + 評估權重 |
| `training_data_means.csv` | 81 × 5,128 | 每個擾動條件的平均表達量（含對照組）|
| `pert_ids_all.csv` | 120 筆 | 測試基因（60 驗證 + 60 測試）|
| `sample_submission.csv` | 120 × 5,128 | 提交格式模板 |

---

## 三、整體解題策略

採用三階段方法，由簡到繁，逐步提升預測品質：

```
[Phase 1] KNN 基線 ─────────────────────────────→ submission_knn.csv
[Phase 2] GNN 深度學習（5折交叉驗證）──────────→ submission_gnn.csv
[Phase 3] Ensemble 混合（GNN×0.7 + KNN×0.3）──→ submission_ensemble.csv（最終）
```

---

## 四、Phase 1：KNN 基線

### 核心概念
**基因功能相似 → 敲低後效果也相似**。
若測試基因 X 和訓練基因 A 在生物功能上很接近，就用 A 的 DE 來預測 X 的 DE。

### 相似度來源（兩層）

**第一層：STRING 蛋白質互動分數**
STRING 是公開的生物資料庫，收錄了人類基因之間的功能性互動關係，分數範圍 0–1000。
- 分數高（≥ 400）= 生物學上確認的功能關聯（同一代謝路徑、蛋白質複合體等）
- 用門檻 400 確保有足夠的連結數量

**第二層：共表現相關係數（後備）**
若測試基因在 STRING 找不到連結，改用 Pearson 相關係數：
- 從對照組細胞（non-targeting）提取每個基因在所有細胞的表達向量
- |r| 越大 = 兩個基因在正常細胞中同漲同跌 = 可能功能相近

### 預測方式
```
1. 計算測試基因 vs 80個訓練基因的相似度
2. 取 Top-5 最相似的鄰居
3. 用 softmax(相似度 / 溫度) 計算加權係數
4. 加權平均鄰居的 DE 向量
5. 裁剪到訓練集的 min/max 範圍內
```

### 優缺點
| | |
|--|--|
| 優點 | 不需要訓練，可立即得到第一份 submission；可解釋性強 |
| 缺點 | 無法捕捉複雜的非線性關係；依賴 STRING 資料庫的完整性 |

---

## 五、Phase 2：GNN 深度學習模型

### 5.1 基因互動圖的建構

GNN 需要一張圖，每個節點是一個基因，邊代表基因之間的關係。

**節點集合**：
```
5,127 個輸出基因 ∪ 80 個訓練擾動基因 ∪ 120 個測試擾動基因
```

**邊的來源（兩種）**：

| 邊類型 | 來源 | 門檻 |
|--------|------|------|
| STRING PPI 邊 | STRING 資料庫 | 分數 ≥ 700（高可信度） |
| 共表現邊 | 對照組細胞的 Pearson 相關 | \|r\| ≥ 0.4，每個基因最多 20 條邊 |

> KNN 用 400 的低門檻（要多點連結），GNN 用 700 的高門檻（只要最可靠的邊）。

### 5.2 節點特徵設計（119 維）

每個基因節點有 3 組特徵，拼接在一起：

**① 表達統計量（5D）**
從對照組細胞計算每個基因的基本統計：
- mean（平均表達量）
- log1p_mean（對數變換後的平均）
- var（方差）
- dispersion（離散度）
- detect_rate（在多少比例的細胞中有表達）

**② 共表現 PCA（50D）**
- 取對照組細胞的表達矩陣，轉置後做 TruncatedSVD
- 每個基因得到 50 維的座標，反映它在「共表現空間」中的位置
- 座標相近的基因 = 共表現模式相似

**③ GO Term 嵌入（64D）**
- Gene Ontology 是一個標準的基因功能標籤體系（分子功能、生物過程、細胞成分）
- 建立基因 × GO term 的二元矩陣（有沒有這個功能標籤）
- 做 TruncatedSVD 得到 64 維嵌入
- 功能相似的基因在此空間中位置相近

```
最終：[5D 表達統計] + [50D 共表現PCA] + [64D GO嵌入] = 119D
      → 透過 MLP 投影到 128D 供 GNN 使用
```

### 5.3 模型架構（GEARS-style 兩階段 GAT）

```
輸入：基因互動圖（節點特徵 119D）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 1：Context Encoder（與擾動無關）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  119D → 投影 MLP → 128D
  → GAT Layer 1 (2 heads)
  → GAT Layer 2 (2 heads)
  → h_base（每個基因的「背景嵌入」，128D）

  [這一步每個 epoch 只做一次，可複用]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Stage 2：Perturbation Propagator（針對特定擾動）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  pert_signal = MLP(h_base[被敲低基因的 index])
                          ↓
  所有節點的輸入 = concat(h_base, pert_signal) → 投影 128D
  → GAT Layer 1 + 軟殘差連接 (h + h_base × 0.1)
  → GAT Layer 2 + 軟殘差連接
  → 只取 5,127 個輸出基因的嵌入
  → 線性解碼器 → 5,127 個 DE 預測值

輸出：每個基因的 DE 值（5,127 個）
```

**GAT（Graph Attention Network）的作用**：
每條邊有注意力權重，讓模型學習「哪些鄰居基因的影響更重要」，而不是對所有鄰居等權平均。

**零樣本泛化的關鍵設計**：
擾動信號從被敲低基因的**圖嵌入**導出，而非查表。
測試基因雖然沒有訓練標籤，但它們的圖嵌入由圖結構和節點特徵決定，模型能對任何在圖上的基因計算擾動信號。

### 5.4 訓練流程（5 折交叉驗證）

```
80 個訓練基因 → 分成 5 折
每折：64個訓練 + 16個驗證

每個 epoch：
  1. DropEdge（隨機丟棄 10% 的邊）→ 防止對特定圖結構過擬合
  2. 對訓練基因按批次處理（batch_size=16）
  3. 每批重新執行 Stage 1（encode_graph）+ Stage 2（forward_perturbation）
  4. 計算損失：前 50 epochs 用 Huber Loss（穩健），之後改用 Weighted MAE
  5. 加入 Label Noise（在 GT 上加微小高斯雜訊 σ=0.001）
  6. 梯度裁剪（max norm=1.0）+ Adam 優化器
  7. Cosine Annealing 調整學習率（1e-3 → 1e-5）

每 10 epochs：
  - 在驗證折上計算 Weighted MAE
  - 若改善則儲存 checkpoint（fold{k}_best.pt）
  - 若連續 10 次（100 epochs）無改善則 Early Stopping
```

**超參數設定**：

| 參數 | 值 | 說明 |
|------|-----|------|
| GNN 隱藏維度 | 128 | 本機版（GPU 版可用 256）|
| GAT 層數 | 2 | 本機版（GPU 版可用 3）|
| Attention Heads | 2 | 本機版（GPU 版可用 4）|
| Dropout | 0.2 | 防止過擬合 |
| Max Epochs | 100 | 本機版（完整訓練 500）|
| Early Stopping | 10次驗證無改善 | 約 100 epochs |

---

## 六、Phase 3：Ensemble 混合

### 概念
兩種方法各有優劣，混合通常優於單一方法：
- **KNN**：直接用生物相似度，對於有明確 STRING 連結的基因較準
- **GNN**：能捕捉複雜非線性關係，但訓練樣本少（只有 80 個），可能有擬合不足

### 混合公式
```
最終預測 = 0.7 × GNN預測 + 0.3 × KNN預測
```

### Alpha 最佳化
可以用網格搜尋（alpha 從 0.0 到 1.0，步長 0.05）在 OOF 預測上找最佳比例：
```python
python src/ensemble.py --optimise-alpha
```

---

## 七、完整執行流程

```bash
# 環境安裝
pip install -r requirements.txt

# Step 1: 下載外部資料（一次性）
python3 src/graph_builder.py --download-string   # STRING PPI 資料庫
python3 src/graph_builder.py --download-go       # Gene Ontology 資料

# Step 2: KNN 基線（快速，不需訓練）
python3 src/knn_baseline.py
# → outputs/submissions/submission_knn.csv

# Step 3: 建立節點特徵快取
python3 src/node_features.py
# → outputs/cache/node_features.npy

# Step 4: GNN 訓練（5折交叉驗證）
python3 src/train.py
# → outputs/checkpoints/fold{1-5}_best.pt
# → outputs/figures/training_curves.png

# Step 5: GNN 預測
python3 src/predict.py
# → outputs/submissions/submission_gnn.csv

# Step 6: Ensemble 混合（最終提交）
python3 src/ensemble.py
# → outputs/submissions/submission_ensemble.csv
```

---

## 八、輸出文件說明

### 模型 Checkpoints
```
outputs/checkpoints/
├── fold1_best.pt   # 第1折最佳模型權重
├── fold2_best.pt
├── fold3_best.pt
├── fold4_best.pt
└── fold5_best.pt
```

### 提交文件
```
outputs/submissions/
├── submission_knn.csv       # Phase 1 KNN 預測（120 × 5127）
├── oof_gnn.csv              # GNN 訓練集的 Out-of-Fold 預測（用於除錯）
├── submission_gnn.csv       # Phase 2 GNN 預測（120 × 5127）
└── submission_ensemble.csv  # Phase 3 最終混合預測（最佳提交）
```

### 分析圖表
```
outputs/figures/
├── 01_de_distribution.png      # DE 值分佈（大多接近 0）
├── 02_pert_difficulty.png      # 各擾動的預測難度
├── 03_de_clustermap.png        # 80個擾動 DE 矩陣聚類圖
├── 04_weight_distribution.png  # 評估權重分佈
├── 05_umap_perts.png           # 擾動 DE 向量的 2D 視覺化
├── 06_string_network.png       # 基因互動網絡圖
├── 07_gene_similarity_heatmap.png  # 訓練基因相似度矩陣
├── 08_cell_distribution.png    # 各擾動條件的細胞數分佈
├── 11_submission_distributions.png # 各 submission 預測值分佈比較
└── training_curves.png         # GNN 5折訓練 Loss & Val WMAE 曲線
```

---

## 九、評估基準與預期結果

| 方法 | 預期 WMAE | 說明 |
|------|-----------|------|
| 零預測（全部 0） | 0.1413 | 最差基準 |
| 官方 baseline（平均DE）| 0.1268 | 競賽官方基準，需要打敗這個 |
| KNN（STRING + 共表現）| < 0.1268 | 目標：比官方 baseline 好 |
| GNN（5折 OOF）| 視訓練結果 | 用 training_curves.png 判斷 |
| Ensemble | 最佳 | 混合兩者取長補短 |

---

## 十、設計亮點總結

1. **零樣本泛化**：GNN 不用查表，而是從圖嵌入導出擾動信號，對任何基因都能預測。

2. **多層次生物知識整合**：
   - STRING 資料庫（蛋白質互動）
   - 單細胞共表現（直接從資料計算）
   - Gene Ontology（人工整理的功能標籤）

3. **損失函數與評估指標一致**：直接優化 Weighted MAE，和競賽評估指標完全一樣。

4. **雙重資料增強**：DropEdge（圖增強）+ Label Noise（標籤增強），在只有 80 個訓練樣本的情況下防止過擬合。

5. **軟殘差連接**：Stage 2 的 `h = h + h_base × 0.1`，讓擾動傳播時保留基礎的基因特性，防止特徵崩塌。
