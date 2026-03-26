import 'board_state.dart';

/// 將 BoardState 匯出為 SGF 格式字串
class SgfExport {
  /// 將棋盤狀態轉為 SGF 字串
  ///
  /// SGF 座標系統：column a-s (左→右)，row a-s (上→下)
  static String toSgf(BoardState board, {String? comment}) {
    final sb = StringBuffer();
    sb.write('(;GM[1]FF[4]CA[UTF-8]');
    sb.write('SZ[${board.rows == board.cols ? board.rows : '${board.cols}:${board.rows}'}]');
    sb.write('AP[BoardScanner:0.1]');

    if (comment != null) {
      sb.write('C[${_escapeSgf(comment)}]');
    }

    // 收集黑子和白子座標
    final blackStones = <String>[];
    final whiteStones = <String>[];

    for (int r = 0; r < board.rows; r++) {
      for (int c = 0; c < board.cols; c++) {
        final stone = board.getStone(r, c);
        if (stone == StoneColor.empty) continue;

        final coord = _toSgfCoord(c, r);
        if (stone == StoneColor.black) {
          blackStones.add(coord);
        } else {
          whiteStones.add(coord);
        }
      }
    }

    if (blackStones.isNotEmpty) {
      sb.write('AB');
      for (final s in blackStones) {
        sb.write('[$s]');
      }
    }

    if (whiteStones.isNotEmpty) {
      sb.write('AW');
      for (final s in whiteStones) {
        sb.write('[$s]');
      }
    }

    sb.write(')');
    return sb.toString();
  }

  /// (col, row) → SGF 座標字串，如 "dp"
  static String _toSgfCoord(int col, int row) {
    return String.fromCharCode(97 + col) + String.fromCharCode(97 + row);
  }

  /// 跳脫 SGF 特殊字元
  static String _escapeSgf(String text) {
    return text.replaceAll('\\', '\\\\').replaceAll(']', '\\]');
  }
}
