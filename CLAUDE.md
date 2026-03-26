# Go Board Recognition

圍棋棋盤辨識 monorepo，從相機/照片辨識棋盤狀態。

## 結構

- `packages/core/` — 純辨識邏輯（Dart + opencv_dart），零 UI 依賴
- `apps/android/` — Android 獨立驗證 app（拍照 → 辨識 → debug overlay）

## 規則

- 所有文件使用繁體中文
- GitHub 操作一律使用 justmaker 身分
- `packages/core/` 不可依賴 Flutter UI（只能用 `dart:*` + opencv_dart）
- 測試優先：改演算法前先加對應測試

## 常用指令

```bash
# Bootstrap（安裝所有 package 依賴）
melos bootstrap

# 跑所有測試
melos run test

# 靜態分析
melos run analyze
```
