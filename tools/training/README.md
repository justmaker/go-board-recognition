# 訓練資料生成工具

從 SGF 棋譜自動生成 CNN 棋子分類器的訓練資料。

## 快速開始

```bash
# 建立虛擬環境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 生成 dataset（含 augmentation + 類別平衡）
python generate_dataset.py \
    --sgf-dir sgf_data \
    --output output/dataset \
    --samples-per-game 5 \
    --augment \
    --balance

# 測試渲染
python board_renderer.py  # 輸出到 output/test_*.png
```

## Pipeline

```
SGF 棋譜 → sgf_parser.py（解析 + 逐手回放）
         → board_renderer.py（渲染成圖片，3 種風格）
         → patch_extractor.py（裁 32×32 交叉點 patches）
         → generate_dataset.py（主 script，整合以上 + split）
```

## 渲染風格

| 風格 | 說明 |
|------|------|
| `wood` | 木紋棋盤，立體感棋子 |
| `app` | App 截圖風格，扁平棋子 |
| `dark` | 深色主題 |

## 輸出結構

```
output/dataset/
├── train/
│   ├── black/    ← 黑子 patches
│   ├── white/    ← 白子 patches
│   └── empty/    ← 空交叉點 patches
├── val/
│   ├── black/
│   ├── white/
│   └── empty/
└── test/
    ├── black/
    ├── white/
    └── empty/
```

## SGF 棋譜來源

`sgf_data/` 中包含從 OGS 下載的公開棋譜。如需更多資料：
- [OGS](https://online-go.com/) — 免費線上對弈
- [KGS Archives](https://www.u-go.net/gamerecords/) — 歷史棋譜
- [Go4Go](https://www.go4go.net/) — 職業棋譜

## 參數說明

```
--sgf-dir          SGF 棋譜目錄
--output           輸出目錄
--samples-per-game 每盤棋取幾個棋盤狀態（預設 10）
--styles           渲染風格，逗號分隔（預設 wood,app,dark）
--augment          啟用 data augmentation
--balance          平衡三類數量（undersample 多數類）
--split            train/val/test 比例（預設 80/10/10）
--seed             隨機種子
```
