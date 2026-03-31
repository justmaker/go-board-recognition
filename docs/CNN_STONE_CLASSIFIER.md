# CNN 棋子分類器 — 開發計畫

## 目標

用 CNN 取代現有的 HSV V-distance + Otsu 棋子分類，提升辨識準確率。

現有方案的問題：
- 白子與棋盤底色 V 值太接近（尤其螢幕翻拍）→ 白子幾乎全部漏掉
- 傳統 CV 天花板約 70-80%

目標準確率：**95%+**

## 架構

```
現有 pipeline（保留）          新增
─────────────────────          ────
圖片 → 棋盤偵測 → 透視校正     （不動）
     → 格線偵測                （不動）
     → [棋子偵測]              ← 換成 CNN
       HSV + Otsu（舊）           交叉點 patch → CNN 三分類
```

只替換 Step 3（棋子偵測），Step 1（棋盤偵測）和 Step 2（格線偵測）不動。

## Model 規格

| 項目 | 規格 |
|------|------|
| 輸入 | 32×32×3 RGB patch（交叉點為中心裁切） |
| 輸出 | 3 類：black / white / empty |
| 架構 | MobileNet-tiny 或自訂小型 CNN |
| 大小目標 | < 1MB（TFLite 量化後） |
| 推論速度 | < 1ms per patch（手機端） |

## 訓練資料

### 來源 1：SGF 棋譜合成（主要）

1. 從公開 SGF 棋譜庫下載棋譜（KGS、OGS）
2. 用不同風格渲染棋盤圖片：
   - 木紋棋盤（實拍風格）
   - App 截圖風格（野狐、弈城、OGS）
   - 純色背景
3. 在每個交叉點裁出 32×32 patch
4. 自動標記 black/white/empty

### 來源 2：螢幕截圖（輔助）

- 從不同圍棋 App 截圖
- 交叉點位置可精確計算（棋盤對齊已知）
- 零標註成本

### Data Augmentation

- 隨機亮度 ±20%
- 隨機對比度 ±15%
- 高斯雜訊
- 輕微旋轉 ±5°
- 輕微模糊（模擬手機拍照）

### 預估資料量

- 10 盤棋 × 200 手 × 361 交叉點 = ~720,000 patches
- 配合 augmentation → 數百萬級訓練樣本
- 目標：至少 50,000 patches（含平衡的三類分佈）

## 實作步驟

### Phase 1：資料準備 `tools/training/`

- [ ] SGF parser（讀取棋譜，逐手回放棋盤狀態）
- [ ] 棋盤渲染器（棋盤狀態 → 圖片，多種風格）
- [ ] Patch 裁切器（圖片 + 交叉點座標 → 32×32 patches）
- [ ] 資料集管理（train/val/test split，類別平衡）

### Phase 2：Model 訓練 `tools/training/`

- [ ] 定義 CNN 架構
- [ ] 訓練 script（PyTorch 或 TensorFlow）
- [ ] 評估 script（confusion matrix、per-class accuracy）
- [ ] 轉 TFLite（量化）

### Phase 3：整合到 Flutter App

- [ ] TFLite model 放入 `packages/core/assets/`
- [ ] 新增 `StoneClassifierCNN` class
- [ ] `BoardRecognition` 的 `_detectStones()` 改用 CNN
- [ ] A/B 切換開關（保留舊演算法作為 fallback）

## 目錄結構

```
go-board-recognition/
├── tools/
│   └── training/
│       ├── README.md
│       ├── requirements.txt      ← Python dependencies
│       ├── sgf_parser.py         ← SGF 棋譜解析
│       ├── board_renderer.py     ← 棋盤圖片渲染
│       ├── patch_extractor.py    ← 交叉點 patch 裁切
│       ├── train.py              ← 訓練 script
│       ├── evaluate.py           ← 評估 script
│       ├── export_tflite.py      ← 轉 TFLite
│       ├── sgf_data/             ← SGF 棋譜檔案
│       └── output/               ← 生成的 dataset + model
└── packages/core/
    └── assets/
        └── stone_classifier.tflite  ← 最終 model
```

## 技術選型

| 項目 | 選擇 | 理由 |
|------|------|------|
| 訓練框架 | PyTorch | 生態成熟、debug 方便 |
| 推論格式 | TFLite | Flutter 有 `tflite_flutter` 套件 |
| 渲染 | Python + Pillow | 簡單、可控 |
| SGF 解析 | 自寫或 `sgfmill` | 依賴少 |

## 風險

1. **合成資料 vs 真實照片的 domain gap** — 需要加夠多 augmentation
2. **TFLite + Flutter 整合** — `tflite_flutter` 套件可能有平台問題
3. **Model 大小** — 需要嚴格控制，不能讓 APK 暴漲

## 時程估計

| Phase | 工時 |
|-------|------|
| Phase 1：資料準備 | 4hr |
| Phase 2：Model 訓練 | 4hr |
| Phase 3：整合到 App | 6hr |
| **總計** | **~14hr** |
