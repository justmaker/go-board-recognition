import 'dart:math';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'board_state.dart';
import 'recognition_debug_info.dart';

/// 交叉點取樣資料
class _IntersectionSample {
  final int row;
  final int col;
  final double avgV;
  final double avgS;
  final double stdV;

  _IntersectionSample({
    required this.row,
    required this.col,
    required this.avgV,
    required this.avgS,
    required this.stdV,
  });
}

/// 辨識結果，包含棋盤狀態和除錯資訊
class RecognitionResult {
  final BoardState boardState;
  final RecognitionDebugInfo debugInfo;
  /// 透視校正後的影像（可用於 debug overlay）
  final cv.Mat? warpedImage;

  RecognitionResult({
    required this.boardState,
    required this.debugInfo,
    this.warpedImage,
  });
}

/// 日誌回呼類型
typedef LogCallback = void Function(String message);

/// OpenCV 棋盤辨識服務
class BoardRecognition {
  /// 最近一次辨識的除錯資訊
  RecognitionDebugInfo? lastDebugInfo;

  /// 日誌回呼（設為 null 則不輸出）
  LogCallback? onLog;

  /// 是否保留 warped 影像在結果中（用於 debug overlay）
  bool keepWarpedImage;

  // 邊緣分析結果（由 _detectGridLines 設定，供 _detectStones 讀取）
  bool _isPartialBoard = false;
  bool _isTopEdge = true;
  bool _isBottomEdge = true;
  bool _isLeftEdge = true;
  bool _isRightEdge = true;

  BoardRecognition({this.onLog, this.keepWarpedImage = false});

  void _log(String msg) => onLog?.call(msg);

  /// 從影像檔案辨識棋盤狀態
  Future<RecognitionResult> recognizeFromImage(String imagePath) async {
    final img = cv.imread(imagePath);
    if (img.isEmpty) {
      throw Exception('無法讀取影像: $imagePath');
    }

    try {
      // 1. 偵測棋盤邊界並進行透視校正
      final warped = _findAndWarpBoard(img);

      // 2. 偵測格線並推斷棋盤大小（含邊緣分析）
      final (rows, cols, intersections) = _detectGridLines(warped);

      // 3. 在每個交叉點偵測棋子
      final grid = _detectStones(warped, rows, cols, intersections);

      final result = RecognitionResult(
        boardState: BoardState(
          rows: rows,
          cols: cols,
          grid: grid,
          isPartial: _isPartialBoard,
          realEdges: [_isTopEdge, _isBottomEdge, _isLeftEdge, _isRightEdge],
        ),
        debugInfo: lastDebugInfo ?? RecognitionDebugInfo(),
        warpedImage: keepWarpedImage ? warped.clone() : null,
      );

      warped.dispose();
      return result;
    } finally {
      img.dispose();
    }
  }

  /// 從已載入的 cv.Mat 辨識棋盤狀態
  Future<RecognitionResult> recognizeFromMat(cv.Mat img) async {
    final warped = _findAndWarpBoard(img);
    final (rows, cols, intersections) = _detectGridLines(warped);
    final grid = _detectStones(warped, rows, cols, intersections);

    final result = RecognitionResult(
      boardState: BoardState(
        rows: rows,
        cols: cols,
        grid: grid,
        isPartial: _isPartialBoard,
        realEdges: [_isTopEdge, _isBottomEdge, _isLeftEdge, _isRightEdge],
      ),
      debugInfo: lastDebugInfo ?? RecognitionDebugInfo(),
      warpedImage: keepWarpedImage ? warped.clone() : null,
    );

    warped.dispose();
    return result;
  }

  // ============================================================
  // 步驟 1：棋盤偵測與透視校正
  // ============================================================

  cv.Mat _findAndWarpBoard(cv.Mat original) {
    _log('[BoardRecognition] 原始影像: ${original.cols}x${original.rows}');

    // 方法 A：顏色偵測 — 嚴格範圍（實拍棋盤）
    var warped = _findBoardByColor(original, 12, 35, 50, 100);
    if (warped != null) {
      _log('[BoardRecognition] 板面偵測: 顏色 (Strict), warped=${warped.cols}x${warped.rows}');
      return warped;
    }

    // 方法 A2：顏色偵測 — 寬鬆範圍（截圖/螢幕翻拍，色彩較淡）
    warped = _findBoardByColor(original, 8, 42, 15, 50);
    if (warped != null) {
      _log('[BoardRecognition] 板面偵測: 顏色 (Loose), warped=${warped.cols}x${warped.rows}');
      return warped;
    }

    // 方法 B：增強型邊緣偵測（Canny + Dilate + Convex Hull）
    final gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY);
    final blurred = cv.gaussianBlur(gray, (5, 5), 1.0);
    final edges = cv.canny(blurred, 30, 150);

    final kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
    final dilated = cv.dilate(edges, kernel, iterations: 2);
    kernel.dispose();

    var edgeWarped = _findBoardByContours(dilated, original);
    if (edgeWarped != null) {
      _log('[BoardRecognition] 板面偵測: 輪廓 (Convex Hull), warped=${edgeWarped.cols}x${edgeWarped.rows}');
      gray.dispose();
      blurred.dispose();
      edges.dispose();
      dilated.dispose();
      return edgeWarped;
    }

    // 方法 C：Hough Lines (長邊偵測)
    var houghWarped = _findBoardByHoughLines(dilated, original);

    gray.dispose();
    blurred.dispose();
    edges.dispose();
    dilated.dispose();

    if (houghWarped != null) {
      _log('[BoardRecognition] 板面偵測: Hough Lines, warped=${houghWarped.cols}x${houghWarped.rows}');
      return houghWarped;
    }

    _log('[BoardRecognition] 板面偵測: 全部失敗，使用原圖 ${original.cols}x${original.rows}');
    return original.clone();
  }

  cv.Mat? _findBoardByColor(
      cv.Mat original, int hLow, int hHigh, int sLow, int vLow) {
    final hsv = cv.cvtColor(original, cv.COLOR_BGR2HSV);

    final lower =
        cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [hLow, sLow, vLow]);
    final upper =
        cv.Mat.fromList(1, 3, cv.MatType.CV_8UC3, [hHigh, 255, 255]);
    final mask = cv.inRange(hsv, lower, upper);
    hsv.dispose();
    lower.dispose();
    upper.dispose();

    final kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15));
    final closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel);
    final cleaned = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel);
    mask.dispose();
    closed.dispose();
    kernel.dispose();

    final (contours, _) = cv.findContours(
      cleaned,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );
    cleaned.dispose();

    if (contours.isEmpty) {
      _log('[BoardRecognition] 顏色偵測(H=$hLow-$hHigh S>=$sLow V>=$vLow): 無輪廓');
      return null;
    }

    cv.VecPoint? bestContour;
    double maxArea = 0;
    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area > maxArea) {
        maxArea = area;
        bestContour = contour;
      }
    }

    final totalArea = original.rows * original.cols;
    final ratio = maxArea / totalArea;
    if (bestContour == null || ratio < 0.10) {
      _log('[BoardRecognition] 顏色偵測(H=$hLow-$hHigh S>=$sLow V>=$vLow): 面積不足 ${(ratio * 100).toStringAsFixed(1)}%');
      return null;
    }

    final rect = cv.minAreaRect(bestContour);
    final boxPoints = cv.boxPoints(rect);
    final corners = _orderPoints2f(boxPoints);

    return _warpPerspective(original, corners);
  }

  cv.Mat? _findBoardByContours(cv.Mat processed, cv.Mat original) {
    final (contours, hierarchy) = cv.findContours(
      processed,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );

    cv.VecPoint? bestCnt;
    double maxArea = 0;

    for (final contour in contours) {
      final area = cv.contourArea(contour);
      if (area < original.rows * original.cols * 0.1) continue;

      final hullMat = cv.convexHull(contour, returnPoints: true);
      final hull = cv.VecPoint.fromMat(hullMat);
      hullMat.dispose();

      final peri = cv.arcLength(hull, true);
      final approx = cv.approxPolyDP(hull, 0.02 * peri, true);
      hull.dispose();

      if (approx.length == 4 && area > maxArea) {
        bestCnt?.dispose();
        maxArea = area;
        bestCnt = approx;
      } else {
        approx.dispose();
      }
    }

    contours.dispose();
    hierarchy.dispose();

    if (bestCnt == null) return null;

    final corners = _orderPoints(bestCnt);
    bestCnt.dispose();

    return _warpPerspective(
        original,
        corners
            .map((p) => cv.Point2f(p.x.toDouble(), p.y.toDouble()))
            .toList());
  }

  cv.Mat? _findBoardByHoughLines(cv.Mat edges, cv.Mat original) {
    final linesMat = cv.HoughLinesP(edges, 1, pi / 180, 50,
        minLineLength: original.cols * 0.2, maxLineGap: 20);

    if (linesMat.rows < 4) {
      linesMat.dispose();
      return null;
    }

    final horizontals = <cv.Vec4i>[];
    final verticals = <cv.Vec4i>[];

    for (int i = 0; i < linesMat.rows; i++) {
      final line = linesMat.at<cv.Vec4i>(i, 0);
      final p1 = cv.Point(line.val1, line.val2);
      final p2 = cv.Point(line.val3, line.val4);

      final dx = (p2.x - p1.x).abs();
      final dy = (p2.y - p1.y).abs();

      if (dx == 0) {
        verticals.add(line);
        continue;
      }

      final slope = dy / dx;
      if (slope < 0.5) {
        horizontals.add(line);
      } else if (slope > 2.0) {
        verticals.add(line);
      }
    }
    linesMat.dispose();

    if (horizontals.length < 2 || verticals.length < 2) return null;

    horizontals.sort(
        (a, b) => ((a.val2 + a.val4) / 2).compareTo((b.val2 + b.val4) / 2));
    final top = horizontals.first;
    final bottom = horizontals.last;

    verticals.sort(
        (a, b) => ((a.val1 + a.val3) / 2).compareTo((b.val1 + b.val3) / 2));
    final left = verticals.first;
    final right = verticals.last;

    cv.Point2f? intersection(cv.Vec4i l1, cv.Vec4i l2) {
      final x1 = l1.val1.toDouble();
      final y1 = l1.val2.toDouble();
      final x2 = l1.val3.toDouble();
      final y2 = l1.val4.toDouble();
      final x3 = l2.val1.toDouble();
      final y3 = l2.val2.toDouble();
      final x4 = l2.val3.toDouble();
      final y4 = l2.val4.toDouble();

      final d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
      if (d == 0) return null;

      final px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
              (x1 - x2) * (x3 * y4 - y3 * x4)) /
          d;
      final py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
              (y1 - y2) * (x3 * y4 - y3 * x4)) /
          d;
      return cv.Point2f(px, py);
    }

    final tl = intersection(top, left);
    final tr = intersection(top, right);
    final bl = intersection(bottom, left);
    final br = intersection(bottom, right);

    if (tl == null || tr == null || bl == null || br == null) return null;

    return _warpPerspective(original, [tl, tr, bl, br]);
  }

  cv.Mat? _warpPerspective(cv.Mat original, List<cv.Point2f> corners) {
    final cornersVec = cv.VecPoint2f.fromList(corners);
    final sorted = _orderPoints2f(cornersVec);
    cornersVec.dispose();

    final w = max(
      _distance2f(sorted[0], sorted[1]),
      _distance2f(sorted[2], sorted[3]),
    ).toInt();
    final h = max(
      _distance2f(sorted[0], sorted[3]),
      _distance2f(sorted[1], sorted[2]),
    ).toInt();
    final size = max(w, h);

    if (size < 100) return null;

    final srcPoints = cv.VecPoint2f.fromList(sorted);
    final dstPoints = cv.VecPoint2f.fromList([
      cv.Point2f(0, 0),
      cv.Point2f(size.toDouble() - 1, 0),
      cv.Point2f(size.toDouble() - 1, size.toDouble() - 1),
      cv.Point2f(0, size.toDouble() - 1),
    ]);

    final matrix = cv.getPerspectiveTransform2f(srcPoints, dstPoints);
    final warped = cv.warpPerspective(original, matrix, (size, size));

    matrix.dispose();
    srcPoints.dispose();
    dstPoints.dispose();

    return warped;
  }

  List<cv.Point2f> _orderPoints2f(cv.VecPoint2f points) {
    final pts = points.toList();
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];
    final remaining = [pts[1], pts[2]];
    remaining.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    return [tl, remaining[0], br, remaining[1]];
  }

  double _distance2f(cv.Point2f a, cv.Point2f b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
  }

  List<cv.Point> _orderPoints(cv.VecPoint points) {
    final pts = points.toList();
    pts.sort((a, b) => (a.x + a.y).compareTo(b.x + b.y));
    final tl = pts[0];
    final br = pts[3];
    final remaining = [pts[1], pts[2]];
    remaining.sort((a, b) => (a.y - a.x).compareTo(b.y - b.x));
    return [tl, remaining[0], br, remaining[1]];
  }

  // ============================================================
  // 步驟 2：格線偵測（暴力搜尋最佳間距）
  // ============================================================

  (int, int, List<List<cv.Point2f>>) _detectGridLines(cv.Mat warped) {
    final size = warped.rows;

    // 1. 初步 Hough 偵測用於計算旋轉校正
    var gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
    var blurred = cv.gaussianBlur(gray, (3, 3), 0);
    var edges = cv.canny(blurred, 50, 150);

    var linesMat = cv.HoughLinesP(
      edges,
      1,
      pi / 180,
      50,
      minLineLength: warped.cols * 0.1,
      maxLineGap: warped.cols / 10,
    );

    // 計算旋轉角度
    final angles = <double>[];
    if (linesMat.rows > 0) {
      for (int i = 0; i < linesMat.rows; i++) {
        final line = linesMat.at<cv.Vec4i>(i, 0);
        final x1 = line.val1.toDouble();
        final y1 = line.val2.toDouble();
        final x2 = line.val3.toDouble();
        final y2 = line.val4.toDouble();
        final dx = x2 - x1;
        final dy = y2 - y1;
        if (dx == 0) continue;
        final angle = atan(dy / dx);

        if (angle.abs() < pi / 6) {
          angles.add(angle);
        } else if (angle.abs() > pi / 3) {
          if (angle > 0) {
            angles.add(angle - pi / 2);
          } else {
            angles.add(angle + pi / 2);
          }
        }
      }
    }
    linesMat.dispose();

    // 應用旋轉校正
    if (angles.isNotEmpty) {
      angles.sort();
      final medianAngle = angles[angles.length ~/ 2];
      if (medianAngle.abs() > 0.005) {
        _log('[BoardRecognition] 應用旋轉校正: ${(medianAngle * 180 / pi).toStringAsFixed(2)}度');
        final center = cv.Point2f(size / 2, size / 2);
        final rotMat =
            cv.getRotationMatrix2D(center, medianAngle * 180 / pi, 1.0);
        final rotated = cv.warpAffine(warped, rotMat, (size, size),
            borderMode: cv.BORDER_REPLICATE);
        rotated.copyTo(warped);
        rotMat.dispose();
        rotated.dispose();

        gray.dispose();
        blurred.dispose();
        edges.dispose();
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY);
        blurred = cv.gaussianBlur(gray, (3, 3), 0);
        edges = cv.canny(blurred, 50, 150);
      }
    }

    // === 正式 Hough 線偵測 ===
    linesMat = cv.HoughLinesP(
      edges,
      1,
      pi / 180,
      50,
      minLineLength: warped.cols * 0.1,
      maxLineGap: warped.cols / 10,
    );

    blurred.dispose();
    edges.dispose();

    final horizontalYs = <double>[];
    final verticalXs = <double>[];

    for (int i = 0; i < linesMat.rows; i++) {
      final line = linesMat.at<cv.Vec4i>(i, 0);
      final x1 = line.val1.toDouble();
      final y1 = line.val2.toDouble();
      final x2 = line.val3.toDouble();
      final y2 = line.val4.toDouble();
      final angle = atan2((y2 - y1).abs(), (x2 - x1).abs());
      if (angle < pi / 6) {
        horizontalYs.add((y1 + y2) / 2);
      } else if (angle > pi / 3) {
        verticalXs.add((x1 + x2) / 2);
      }
    }
    linesMat.dispose();

    final hClusters = _clusterValues(horizontalYs, size * 0.02);
    final vClusters = _clusterValues(verticalXs, size * 0.02);

    // === 投影法 dip 偵測 ===
    final hProj = _computeProjection(gray, true);
    final vProj = _computeProjection(gray, false);
    gray.dispose();

    var k = max(3, size ~/ 200);
    if (k % 2 == 0) k++;
    final hSmooth = _smoothProfile(hProj, k);
    final vSmooth = _smoothProfile(vProj, k);

    final minDist = size ~/ 30;
    final hDips = _findDips(hSmooth, minDist);
    final vDips = _findDips(vSmooth, minDist);

    // === 合併 Hough + 投影 ===
    final hCombined = _combinePositions(hDips, hClusters, size * 0.025);
    final vCombined = _combinePositions(vDips, vClusters, size * 0.025);

    // === 過濾影像邊緣位置 ===
    final edgeMargin = size * 0.05;
    final hFiltered = hCombined
        .where((p) => p >= edgeMargin && p <= size - edgeMargin)
        .toList();
    final vFiltered = vCombined
        .where((p) => p >= edgeMargin && p <= size - edgeMargin)
        .toList();

    _log('[BoardRecognition] Hough: H=${hClusters.length}, V=${vClusters.length}');
    _log('[BoardRecognition] Dips: H=${hDips.length}, V=${vDips.length}');
    _log('[BoardRecognition] Combined: H=${hCombined.length}, V=${vCombined.length}');
    _log('[BoardRecognition] Edge filtered: H=${hFiltered.length}, V=${vFiltered.length}');

    // === 暴力搜尋 H/V 各自的最佳間距和相位 ===
    var (hSpacing, hPhase, hInl) =
        _findBestSpacing(hFiltered, size.toDouble());
    var (vSpacing, vPhase, vInl) =
        _findBestSpacing(vFiltered, size.toDouble());

    // Cross-validate
    final spacingDiff =
        (hSpacing - vSpacing).abs() / max(hSpacing, vSpacing);
    if (spacingDiff > 0.15) {
      if (hInl >= vInl) {
        final (newPhase, newInl) = _findBestPhase(vFiltered, hSpacing);
        vSpacing = hSpacing;
        vPhase = newPhase;
        vInl = newInl;
        _log('[BoardRecognition] V 間距修正: 使用 H spacing $hSpacing');
      } else {
        final (newPhase, newInl) = _findBestPhase(hFiltered, vSpacing);
        hSpacing = vSpacing;
        hPhase = newPhase;
        hInl = newInl;
        _log('[BoardRecognition] H 間距修正: 使用 V spacing $vSpacing');
      }
    }

    var hLines =
        _generateGridFromSpacing(hPhase, hSpacing, size.toDouble());
    var vLines =
        _generateGridFromSpacing(vPhase, vSpacing, size.toDouble());

    // === 邊緣留白分析：判斷是完整棋盤還是局部盤面 ===
    final spacing = hInl >= vInl ? hSpacing : vSpacing;

    final topMargin = hLines.first;
    final bottomMargin = size - hLines.last;
    final leftMargin = vLines.first;
    final rightMargin = size - vLines.last;

    // 邊距 > 間距 * 0.3 → 真正的棋盤邊（有外圍留白）
    const realEdgeRatio = 0.3;

    _isTopEdge = topMargin > spacing * realEdgeRatio;
    _isBottomEdge = bottomMargin > spacing * realEdgeRatio;
    _isLeftEdge = leftMargin > spacing * realEdgeRatio;
    _isRightEdge = rightMargin > spacing * realEdgeRatio;

    final allEdgesReal =
        _isTopEdge && _isBottomEdge && _isLeftEdge && _isRightEdge;

    int rows;
    int cols;

    if (allEdgesReal) {
      // 完整棋盤：snap 到標準大小 9/13/19
      _isPartialBoard = false;
      final hRatio = size / hSpacing;
      final vRatio = size / vSpacing;

      int snapToStandard(double ratio) {
        if (ratio < 11.5) return 9;
        if (ratio < 17) return 13;
        return 19;
      }

      rows = snapToStandard(hRatio);
      cols = snapToStandard(vRatio);
    } else {
      // 局部盤面：只保留有偵測證據範圍內的格線
      _isPartialBoard = true;

      // 用偵測到的實際位置限制範圍（加半格緩衝）
      hLines = _clipToEvidence(hLines, hFiltered, hSpacing);
      vLines = _clipToEvidence(vLines, vFiltered, vSpacing);

      rows = hLines.length;
      cols = vLines.length;
    }

    _log('[BoardRecognition] 邊距: top=${topMargin.toStringAsFixed(1)}, '
        'bottom=${bottomMargin.toStringAsFixed(1)}, '
        'left=${leftMargin.toStringAsFixed(1)}, '
        'right=${rightMargin.toStringAsFixed(1)} '
        '(spacing=${spacing.toStringAsFixed(1)})');
    _log('[BoardRecognition] 邊緣: top=${_isTopEdge}, bottom=${_isBottomEdge}, '
        'left=${_isLeftEdge}, right=${_isRightEdge} '
        '→ partial=$_isPartialBoard');

    final List<double> hFinal;
    final List<double> vFinal;

    if (_isPartialBoard) {
      // 局部盤面：用偵測到的格線，微調對齊 projection
      hFinal = _refineToProjection(hLines, hSmooth, hSpacing);
      vFinal = _refineToProjection(vLines, vSmooth, vSpacing);
    } else {
      // 完整棋盤：trimToSize 再 refine
      final hTrimmed =
          _trimToSize(hLines, rows, hSpacing, size.toDouble());
      final vTrimmed =
          _trimToSize(vLines, cols, vSpacing, size.toDouble());
      hFinal = _refineToProjection(hTrimmed, hSmooth, hSpacing);
      vFinal = _refineToProjection(vTrimmed, vSmooth, vSpacing);
    }

    final intersections = <List<cv.Point2f>>[];
    for (int r = 0; r < rows; r++) {
      final row = <cv.Point2f>[];
      for (int c = 0; c < cols; c++) {
        row.add(cv.Point2f(vFinal[c], hFinal[r]));
      }
      intersections.add(row);
    }

    _log('[BoardRecognition] 間距: H=${hSpacing.toStringAsFixed(1)} (inl=$hInl), V=${vSpacing.toStringAsFixed(1)} (inl=$vInl)');
    _log('[BoardRecognition] 格線: ${rows}x$cols (partial=$_isPartialBoard)');

    return (rows, cols, intersections);
  }

  (double, int) _findBestPhase(List<double> positions, double spacing) {
    final tolerance = spacing * 0.12;
    var bestPhase = 0.0;
    var bestInliers = 0;
    for (final ref in positions) {
      final phase = ref % spacing;
      var inliers = 0;
      for (final p in positions) {
        var remainder = (p - phase) % spacing;
        if (remainder > spacing / 2) remainder = spacing - remainder;
        if (remainder < tolerance) inliers++;
      }
      if (inliers > bestInliers) {
        bestInliers = inliers;
        bestPhase = phase;
      }
    }
    return (bestPhase, bestInliers);
  }

  (double, double, int) _findBestSpacing(
      List<double> positions, double totalSize) {
    if (positions.length < 3) {
      return (totalSize / 14, totalSize * 0.05, 0);
    }

    final minSp = (totalSize * 0.04).toInt();
    final maxSp = (totalSize * 0.12).toInt();

    var bestSpacing = totalSize / 14;
    var bestPhase = 0.0;
    var bestScore = 0.0;

    for (int sp = minSp; sp <= maxSp; sp++) {
      final tolerance = sp * 0.12;
      final spSqrt = sqrt(sp.toDouble());

      final ratio = totalSize / sp;
      var scoreMultiplier = 1.0;
      if (ratio >= 18 && ratio <= 20) {
        scoreMultiplier = 1.2;
      } else if (ratio >= 12 && ratio <= 14) {
        scoreMultiplier = 1.1;
      } else if (ratio >= 8 && ratio <= 10) {
        scoreMultiplier = 1.1;
      }

      for (final ref in positions) {
        final phase = ref % sp;
        var inliers = 0;
        for (final p in positions) {
          var remainder = (p - phase) % sp;
          if (remainder > sp / 2) remainder = sp - remainder;
          if (remainder < tolerance) inliers++;
        }
        var score = inliers * spSqrt * scoreMultiplier;

        if (score > bestScore) {
          bestScore = score;
          bestSpacing = sp.toDouble();
          bestPhase = phase;
        }
      }
    }

    // 用 inlier 位置精修間距
    final tolerance = bestSpacing * 0.12;
    final inlierPos = <double>[];
    for (final p in positions) {
      var remainder = (p - bestPhase) % bestSpacing;
      if (remainder > bestSpacing / 2) remainder = bestSpacing - remainder;
      if (remainder < tolerance) inlierPos.add(p);
    }
    inlierPos.sort();

    if (inlierPos.length >= 2) {
      final refinedDiffs = <double>[];
      for (int i = 0; i < inlierPos.length - 1; i++) {
        final d = inlierPos[i + 1] - inlierPos[i];
        final n = (d / bestSpacing).round();
        if (n > 0) refinedDiffs.add(d / n);
      }
      if (refinedDiffs.isNotEmpty) {
        bestSpacing =
            refinedDiffs.reduce((a, b) => a + b) / refinedDiffs.length;
      }
    }

    // 用精修間距重算最佳相位
    final tol2 = bestSpacing * 0.12;
    var bestPhase2 = 0.0;
    var bestInliers2 = 0;
    for (final ref in positions) {
      final phase = ref % bestSpacing;
      var inliers = 0;
      for (final p in positions) {
        var remainder = (p - phase) % bestSpacing;
        if (remainder > bestSpacing / 2) remainder = bestSpacing - remainder;
        if (remainder < tol2) inliers++;
      }
      if (inliers > bestInliers2) {
        bestInliers2 = inliers;
        bestPhase2 = phase;
      }
    }

    return (bestSpacing, bestPhase2, bestInliers2);
  }

  List<double> _generateGridFromSpacing(
      double phase, double spacing, double totalSize) {
    final lines = <double>[];
    var k = 0;
    while (phase + k * spacing >= 0) {
      k--;
    }
    k++;
    while (phase + k * spacing < totalSize) {
      lines.add(phase + k * spacing);
      k++;
    }
    return lines;
  }

  /// 局部盤面用：只保留有偵測證據範圍內的格線
  /// generated = 等間距填滿整張圖的格線
  /// evidence = 實際偵測到的位置（Hough + projection dips）
  /// 保留範圍：evidence 最小值 - spacing/2 到最大值 + spacing/2
  List<double> _clipToEvidence(
      List<double> generated, List<double> evidence, double spacing) {
    if (evidence.isEmpty) return generated;
    final lo = evidence.first - spacing * 0.5;
    final hi = evidence.last + spacing * 0.5;
    return generated.where((p) => p >= lo && p <= hi).toList();
  }

  List<double> _trimToSize(
      List<double> lines, int target, double spacing, double totalSize) {
    final result = List<double>.from(lines);
    if (result.length > target) {
      final start = (result.length - target) ~/ 2;
      return result.sublist(start, start + target);
    }
    while (result.length < target) {
      final nextP = result.last + spacing;
      final prevP = result.first - spacing;
      if (nextP < totalSize) {
        result.add(nextP);
      } else if (prevP >= 0) {
        result.insert(0, prevP);
      } else if (nextP - totalSize < result.first.abs()) {
        result.add(nextP);
      } else {
        result.insert(0, prevP);
      }
    }
    return result.sublist(0, target);
  }

  List<double> _refineToProjection(
      List<double> positions, List<double> profile, double spacing) {
    final searchRadius = (spacing * 0.25).toInt();
    var bestShift = 0;
    var bestScore = double.infinity;
    for (int shift = -searchRadius; shift <= searchRadius; shift++) {
      var score = 0.0;
      for (final pos in positions) {
        final idx = (pos + shift).round().clamp(0, profile.length - 1);
        score += profile[idx];
      }
      if (score < bestScore) {
        bestScore = score;
        bestShift = shift;
      }
    }
    return positions.map((p) => p + bestShift).toList();
  }

  List<double> _computeProjection(cv.Mat singleChannel, bool horizontal) {
    final rows = singleChannel.rows;
    final cols = singleChannel.cols;
    final data = singleChannel.data;

    if (horizontal) {
      final proj = List<double>.filled(rows, 0);
      for (int y = 0; y < rows; y++) {
        double sum = 0;
        final offset = y * cols;
        for (int x = 0; x < cols; x++) {
          sum += data[offset + x];
        }
        proj[y] = sum / cols;
      }
      return proj;
    } else {
      final proj = List<double>.filled(cols, 0);
      for (int y = 0; y < rows; y++) {
        final offset = y * cols;
        for (int x = 0; x < cols; x++) {
          proj[x] += data[offset + x];
        }
      }
      for (int x = 0; x < cols; x++) {
        proj[x] /= rows;
      }
      return proj;
    }
  }

  List<double> _smoothProfile(List<double> profile, int kernelSize) {
    final result = List<double>.filled(profile.length, 0);
    final halfK = kernelSize ~/ 2;
    for (int i = 0; i < profile.length; i++) {
      double sum = 0;
      int count = 0;
      final start = max(0, i - halfK);
      final end = min(profile.length - 1, i + halfK);
      for (int j = start; j <= end; j++) {
        sum += profile[j];
        count++;
      }
      result[i] = sum / count;
    }
    return result;
  }

  List<int> _findDips(List<double> profile, int minDist) {
    final sorted = List<double>.from(profile)..sort();
    final median = sorted[sorted.length ~/ 2];

    final dips = <int>[];
    for (int i = minDist; i < profile.length - minDist; i++) {
      double minVal = double.infinity;
      for (int j = max(0, i - minDist);
          j <= min(profile.length - 1, i + minDist);
          j++) {
        if (profile[j] < minVal) minVal = profile[j];
      }
      if (profile[i] == minVal && profile[i] < median) {
        dips.add(i);
      }
    }
    return dips;
  }

  List<double> _combinePositions(
      List<int> dips, List<double> hough, double tolerance) {
    final allPos = <double>[];
    for (final d in dips) {
      allPos.add(d.toDouble());
    }
    allPos.addAll(hough);
    allPos.sort();

    if (allPos.isEmpty) return [];
    final clusters = <double>[];
    var cSum = allPos[0];
    var cCount = 1;
    for (int i = 1; i < allPos.length; i++) {
      if (allPos[i] - allPos[i - 1] < tolerance) {
        cSum += allPos[i];
        cCount++;
      } else {
        clusters.add(cSum / cCount);
        cSum = allPos[i];
        cCount = 1;
      }
    }
    clusters.add(cSum / cCount);
    return clusters;
  }

  List<double> _clusterValues(List<double> values, double threshold) {
    if (values.isEmpty) return [];
    values.sort();
    final clusters = <double>[];
    var clusterSum = values[0];
    var clusterCount = 1;
    for (int i = 1; i < values.length; i++) {
      if (values[i] - values[i - 1] < threshold) {
        clusterSum += values[i];
        clusterCount++;
      } else {
        clusters.add(clusterSum / clusterCount);
        clusterSum = values[i];
        clusterCount = 1;
      }
    }
    clusters.add(clusterSum / clusterCount);
    return clusters;
  }

  // ============================================================
  // 步驟 3：棋子偵測
  // ============================================================

  List<List<StoneColor>> _detectStones(
    cv.Mat warped,
    int rows,
    int cols,
    List<List<cv.Point2f>> intersections,
  ) {
    final debug = RecognitionDebugInfo();
    debug.detectedBoardSize = max(rows, cols);
    debug.detectedRows = rows;
    debug.detectedCols = cols;

    final hsv = cv.cvtColor(warped, cv.COLOR_BGR2HSV);
    final grid = List.generate(
      rows,
      (_) => List.filled(cols, StoneColor.empty),
    );

    final sampleRadius = (warped.cols / cols * 0.35).round();
    final samples = <_IntersectionSample>[];

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        final pt = intersections[r][c];
        final x = pt.x.round();
        final y = pt.y.round();

        // 局部盤面：裁切邊的最外行/列直接跳過（warp 邊界可能有非棋盤內容）
        final skipEdge = _isPartialBoard &&
            ((r == 0 && !_isTopEdge) ||
             (r == rows - 1 && !_isBottomEdge) ||
             (c == 0 && !_isLeftEdge) ||
             (c == cols - 1 && !_isRightEdge));

        if (skipEdge ||
            x < sampleRadius ||
            x >= warped.cols - sampleRadius ||
            y < sampleRadius ||
            y >= warped.rows - sampleRadius) {
          samples.add(_IntersectionSample(
            row: r,
            col: c,
            avgV: -1.0,
            avgS: 0.0,
            stdV: 0.0,
          ));
          continue;
        }

        var totalV = 0.0;
        var totalS = 0.0;
        var totalV2 = 0.0;
        var sampleCount = 0;

        for (int dy = -sampleRadius; dy <= sampleRadius; dy++) {
          for (int dx = -sampleRadius; dx <= sampleRadius; dx++) {
            final sx = (x + dx).clamp(0, warped.cols - 1);
            final sy = (y + dy).clamp(0, warped.rows - 1);
            final pixel = hsv.atPixel(sy, sx);
            totalS += pixel[1];
            final v = pixel[2].toDouble();
            totalV += v;
            totalV2 += v * v;
            sampleCount++;
          }
        }

        final avgV = totalV / sampleCount;
        final avgS = totalS / sampleCount;
        final variance = (totalV2 / sampleCount) - (avgV * avgV);
        final stdV = sqrt(max(0, variance));

        samples.add(_IntersectionSample(
          row: r,
          col: c,
          avgV: avgV,
          avgS: avgS,
          stdV: stdV,
        ));
      }
    }
    hsv.dispose();

    // === V/S 值統計 ===
    final validSamples = samples.where((s) => s.avgV >= 0).toList();
    if (validSamples.isEmpty) return grid;

    final vValues = validSamples.map((s) => s.avgV).toList();
    final sValues = validSamples.map((s) => s.avgS).toList();

    var minV = 255.0;
    var maxV = 0.0;
    for (final v in vValues) {
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    debug.vMin = minV;
    debug.vMax = maxV;

    final sortedS = List<double>.from(sValues)..sort();
    final boardMedianS =
        sortedS.isNotEmpty ? sortedS[sortedS.length ~/ 2] : 0.0;

    // === Mode-based 自適應閾值 ===
    const binSize = 8;
    final bins = List.filled(256 ~/ binSize + 1, 0);
    for (final v in vValues) {
      bins[v.toInt() ~/ binSize]++;
    }
    var maxBin = 0;
    var maxCount = 0;
    for (int i = 0; i < bins.length; i++) {
      if (bins[i] > maxCount) {
        maxCount = bins[i];
        maxBin = i;
      }
    }
    final modeV = (maxBin * binSize + binSize / 2).toDouble();
    debug.clusterCenters = [modeV];

    // 根據實際 V range 縮放門檻：螢幕拍照對比度低，gap 要跟著縮小
    final vRange = maxV - minV;
    final scaleFactor = (vRange / 180.0).clamp(0.35, 1.0);

    var thresholdBB = modeV - 50.0 * scaleFactor;
    var thresholdBW = modeV + 40.0 * scaleFactor;
    thresholdBB = max(thresholdBB, 35.0);
    thresholdBW = min(thresholdBW, 230.0);

    debug.thresholdBlackBoard = thresholdBB;
    debug.thresholdBoardWhite = thresholdBW;

    final satLimitBlack = max(boardMedianS * 2.5, 60.0);
    final satLimitWhite = max(boardMedianS * 1.5, 40.0);
    debug.satLimitBlack = satLimitBlack;
    debug.satLimitWhite = satLimitWhite;

    // stdV 統計：棋子表面均勻，stdV 低；空交叉點有格線，stdV 高
    final sortedStdV = validSamples.map((s) => s.stdV).toList()..sort();
    final medianStdV = sortedStdV[sortedStdV.length ~/ 2];
    final uniformThreshold = medianStdV * 0.75;

    _log('[BoardRecognition] vRange=${vRange.toStringAsFixed(0)}, scale=${scaleFactor.toStringAsFixed(2)}, medianStdV=${medianStdV.toStringAsFixed(1)}');

    // === 分類 ===
    for (final sample in samples) {
      if (sample.avgV < 0) {
        debug.emptyCount++;
        continue;
      }

      final isUniform = sample.stdV < uniformThreshold;

      // 主要判定：超過門檻直接分類
      if (sample.avgV < thresholdBB && sample.avgS < satLimitBlack) {
        grid[sample.row][sample.col] = StoneColor.black;
        debug.blackCount++;
      } else if (sample.avgV > thresholdBW && sample.avgS < satLimitWhite) {
        grid[sample.row][sample.col] = StoneColor.white;
        debug.whiteCount++;
      } else if (isUniform) {
        // 邊界案例：未過門檻但表面均勻 → 用 stdV 輔助判定
        // 均勻 + 明顯偏亮 → 可能是白子
        // 均勻 + 明顯偏暗 → 可能是黑子
        final halfGapWhite = (thresholdBW - modeV) * 0.5;
        final halfGapBlack = (modeV - thresholdBB) * 0.5;
        if (sample.avgV > modeV + halfGapWhite && sample.avgS < satLimitWhite) {
          grid[sample.row][sample.col] = StoneColor.white;
          debug.whiteCount++;
        } else if (sample.avgV < modeV - halfGapBlack && sample.avgS < satLimitBlack) {
          grid[sample.row][sample.col] = StoneColor.black;
          debug.blackCount++;
        } else {
          debug.emptyCount++;
        }
      } else {
        debug.emptyCount++;
      }
    }

    _log('[BoardRecognition] 棋子: 黑=${debug.blackCount}, 白=${debug.whiteCount}, 空=${debug.emptyCount}');
    _log('[BoardRecognition] modeV=${modeV.toStringAsFixed(0)}, BB<${thresholdBB.toStringAsFixed(0)}, BW>${thresholdBW.toStringAsFixed(0)}');

    // 邊緣分析結果寫入 debug info
    debug.isPartialBoard = _isPartialBoard;
    debug.isTopEdge = _isTopEdge;
    debug.isBottomEdge = _isBottomEdge;
    debug.isLeftEdge = _isLeftEdge;
    debug.isRightEdge = _isRightEdge;

    lastDebugInfo = debug;
    return grid;
  }
}
