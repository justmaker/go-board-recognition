# Go Board Recognition 圍棋棋盤辨識

從相機拍照或圖片自動辨識圍棋棋盤狀態，支援 9/13/19 路棋盤及局部盤面。

## ✨ 功能

### 📷 拍照辨識
- **相機即時預覽**：開啟相機後顯示格線 overlay，對齊棋盤後拍照，支援切換 9/13/19 路格線
- **快速拍照**：直接開啟系統相機拍照，跳過預覽
- **相簿選取**：從手機相簿選擇已有的棋盤照片

### 🔍 辨識結果
- 自動偵測棋盤邊界並進行透視校正
- 自動辨識黑子、白子、空交叉點
- 支援完整棋盤 (9×9, 13×13, 19×19) 和局部盤面
- Debug overlay 顯示辨識細節

### ✏️ 手動修正
辨識完成後，點擊右上角 ✏️ 按鈕進入編輯模式：
- 點擊交叉點循環切換：**空 → 黑 → 白 → 空**
- 修正後的結果也能匯出 SGF

### 📤 SGF 匯出
點擊右上角分享按鈕，將辨識結果（或手動修正後的結果）匯出為 [SGF 格式](https://www.red-bean.com/sgf/)，可分享給其他圍棋軟體使用。

### 📜 歷史紀錄
- 每次辨識自動儲存結果（棋盤大小、棋子數、時間）
- 點擊右上角 🕐 按鈕瀏覽過往辨識紀錄

## 📱 安裝

從 [Releases](https://github.com/justmaker/go-board-recognition/releases) 下載最新 APK：

> ⚠️ 目前為 Closed Alpha，僅提供 Android debug APK

## 🏗️ 架構

```
go-board-recognition/
├── packages/core/          ← 辨識核心（純 Dart + opencv_dart，零 UI 依賴）
│   └── lib/src/
│       ├── board_recognition.dart   — 辨識管線主流程
│       ├── board_state.dart         — 棋盤狀態資料模型
│       ├── sgf_export.dart          — SGF 格式匯出
│       └── recognition_debug_info.dart — 除錯資訊
├── apps/android/           ← Flutter Android app
│   └── lib/
│       ├── screens/
│       │   ├── home_screen.dart     — 首頁（拍照/選圖入口）
│       │   ├── camera_screen.dart   — 相機即時預覽 + 格線 overlay
│       │   ├── result_screen.dart   — 辨識結果 + 手動修正 + SGF 匯出
│       │   └── history_screen.dart  — 歷史紀錄瀏覽
│       ├── widgets/
│       │   └── debug_board_painter.dart — 棋盤 debug 繪製
│       └── services/
│           └── history_service.dart — 歷史紀錄儲存服務
└── test_images/            ← 測試用圍棋圖片
```

## 🔧 開發

### 環境需求
- Flutter 3.27+
- Dart SDK 3.6+
- [Melos](https://melos.invertase.dev/) (monorepo 管理)

### 常用指令

```bash
# 安裝依賴
melos bootstrap

# 靜態分析
melos run analyze

# 執行測試
melos run test
```

### 辨識管線

```
圖片
  → 棋盤偵測（顏色遮罩 / 輪廓凸包 / Hough Lines，三段 fallback）
  → 透視校正
  → 旋轉校正
  → 格線偵測（Hough + 灰度投影 dip + 暴力搜尋最佳間距）
  → 棋子偵測（HSV V-distance + Otsu 自適應門檻）
  → BoardState
```

## 📄 授權

MIT
