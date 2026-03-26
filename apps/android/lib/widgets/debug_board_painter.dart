import 'package:flutter/material.dart';
import 'package:go_board_core/go_board_core.dart';

/// 棋盤辨識結果的 debug 繪製器
/// 顯示格線、星位、辨識到的黑白棋子
/// 支援非正方形局部盤面 + 邊緣標示
class DebugBoardPainter extends CustomPainter {
  final BoardState boardState;

  DebugBoardPainter({required this.boardState});

  @override
  void paint(Canvas canvas, Size size) {
    final rows = boardState.rows;
    final cols = boardState.cols;
    if (rows < 2 || cols < 2) return;

    final padding = size.shortestSide * 0.04;
    final boardWidth = size.width - padding * 2;
    final boardHeight = size.height - padding * 2;
    final cellW = boardWidth / (cols - 1);
    final cellH = boardHeight / (rows - 1);
    final stoneRadius = (cellW < cellH ? cellW : cellH) * 0.42;

    // 棋盤底色
    final bgPaint = Paint()..color = const Color(0xFFDEB887);
    canvas.drawRect(Rect.fromLTWH(0, 0, size.width, size.height), bgPaint);

    // 格線
    final linePaint = Paint()
      ..color = Colors.black
      ..strokeWidth = 1.0;

    for (int c = 0; c < cols; c++) {
      final x = padding + c * cellW;
      canvas.drawLine(
        Offset(x, padding),
        Offset(x, padding + boardHeight),
        linePaint,
      );
    }
    for (int r = 0; r < rows; r++) {
      final y = padding + r * cellH;
      canvas.drawLine(
        Offset(padding, y),
        Offset(padding + boardWidth, y),
        linePaint,
      );
    }

    // 局部盤面邊緣標示：非真實邊用虛線標示
    if (boardState.isPartial) {
      final edgePaint = Paint()
        ..color = Colors.red.withValues(alpha: 0.6)
        ..strokeWidth = 3.0
        ..style = PaintingStyle.stroke;

      final edges = boardState.realEdges;
      // [top, bottom, left, right] — 畫非真實邊
      if (!edges[0]) {
        // top is cropped
        canvas.drawLine(
            Offset(0, padding * 0.5), Offset(size.width, padding * 0.5), edgePaint);
      }
      if (!edges[1]) {
        // bottom is cropped
        canvas.drawLine(
            Offset(0, size.height - padding * 0.5),
            Offset(size.width, size.height - padding * 0.5),
            edgePaint);
      }
      if (!edges[2]) {
        // left is cropped
        canvas.drawLine(
            Offset(padding * 0.5, 0), Offset(padding * 0.5, size.height), edgePaint);
      }
      if (!edges[3]) {
        // right is cropped
        canvas.drawLine(
            Offset(size.width - padding * 0.5, 0),
            Offset(size.width - padding * 0.5, size.height),
            edgePaint);
      }
    }

    // 星位（只有完整棋盤才畫）
    if (!boardState.isPartial) {
      final starPoints = _getStarPoints(rows);
      final starPaint = Paint()..color = Colors.black;
      final starRadius = stoneRadius * 0.2;
      for (final sp in starPoints) {
        if (sp.row < rows && sp.col < cols) {
          canvas.drawCircle(
            Offset(padding + sp.col * cellW, padding + sp.row * cellH),
            starRadius,
            starPaint,
          );
        }
      }
    }

    // 棋子
    final blackFill = Paint()..color = Colors.black;
    final whiteFill = Paint()..color = Colors.white;
    final stoneStroke = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        final stone = boardState.getStone(r, c);
        if (stone == StoneColor.empty) continue;

        final center = Offset(padding + c * cellW, padding + r * cellH);

        if (stone == StoneColor.black) {
          canvas.drawCircle(center, stoneRadius, blackFill);
        } else {
          canvas.drawCircle(center, stoneRadius, whiteFill);
          canvas.drawCircle(center, stoneRadius, stoneStroke);
        }
      }
    }

    // 座標標籤
    final labelStyle = TextStyle(
      color: Colors.black54,
      fontSize: (cellW < cellH ? cellW : cellH) * 0.3,
    );
    final tp = TextPainter(textDirection: TextDirection.ltr);

    // 列號（左側）
    for (int r = 0; r < rows; r++) {
      tp.text = TextSpan(text: '${rows - r}', style: labelStyle);
      tp.layout();
      tp.paint(
        canvas,
        Offset(padding * 0.1, padding + r * cellH - tp.height / 2),
      );
    }

    // 行號（下方），跳過 I
    for (int c = 0; c < cols; c++) {
      final letter = String.fromCharCode(
          65 + c + (c >= 8 ? 1 : 0)); // A-H, J-T (跳過 I)
      tp.text = TextSpan(text: letter, style: labelStyle);
      tp.layout();
      tp.paint(
        canvas,
        Offset(padding + c * cellW - tp.width / 2,
            size.height - padding * 0.9),
      );
    }
  }

  /// 取得星位座標
  List<BoardPosition> _getStarPoints(int size) {
    switch (size) {
      case 19:
        return const [
          BoardPosition(3, 3), BoardPosition(3, 9), BoardPosition(3, 15),
          BoardPosition(9, 3), BoardPosition(9, 9), BoardPosition(9, 15),
          BoardPosition(15, 3), BoardPosition(15, 9), BoardPosition(15, 15),
        ];
      case 13:
        return const [
          BoardPosition(3, 3), BoardPosition(3, 9),
          BoardPosition(6, 6),
          BoardPosition(9, 3), BoardPosition(9, 9),
        ];
      case 9:
        return const [
          BoardPosition(2, 2), BoardPosition(2, 6),
          BoardPosition(4, 4),
          BoardPosition(6, 2), BoardPosition(6, 6),
        ];
      default:
        return const [];
    }
  }

  @override
  bool shouldRepaint(DebugBoardPainter oldDelegate) =>
      oldDelegate.boardState != boardState;
}
