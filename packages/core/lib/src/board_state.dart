/// 棋盤上每個交叉點的狀態
enum StoneColor {
  empty,
  black,
  white;

  StoneColor get opponent {
    switch (this) {
      case StoneColor.black:
        return StoneColor.white;
      case StoneColor.white:
        return StoneColor.black;
      case StoneColor.empty:
        return StoneColor.empty;
    }
  }
}

/// 棋盤座標
class BoardPosition {
  final int row;
  final int col;

  const BoardPosition(this.row, this.col);

  @override
  bool operator ==(Object other) =>
      other is BoardPosition && other.row == row && other.col == col;

  @override
  int get hashCode => row * 31 + col;

  @override
  String toString() => '($row, $col)';
}

/// 棋盤狀態資料模型
///
/// 完整棋盤: rows == cols (9x9, 13x13, 19x19)
/// 局部棋盤: rows 和 cols 可能不同
class BoardState {
  final int rows;
  final int cols;
  final List<List<StoneColor>> grid;
  final StoneColor nextPlayer;
  final double komi;

  /// 是否為局部盤面（有邊被裁切）
  final bool isPartial;

  /// 四邊是否為真實棋盤邊（top, bottom, left, right）
  final List<bool> realEdges;

  /// 向後相容：正方形棋盤回傳 rows
  int get boardSize => rows;

  BoardState({
    int? boardSize,
    int? rows,
    int? cols,
    List<List<StoneColor>>? grid,
    this.nextPlayer = StoneColor.black,
    this.komi = 7.5,
    this.isPartial = false,
    List<bool>? realEdges,
  })  : rows = rows ?? boardSize ?? 19,
        cols = cols ?? boardSize ?? 19,
        realEdges = realEdges ?? const [true, true, true, true],
        grid = grid ??
            List.generate(
              rows ?? boardSize ?? 19,
              (_) => List.filled(cols ?? boardSize ?? 19, StoneColor.empty),
            );

  /// 取得指定位置的棋子顏色
  StoneColor getStone(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      throw RangeError(
          'Position ($row, $col) out of bounds for $rows x $cols board');
    }
    return grid[row][col];
  }

  /// 設定指定位置的棋子，回傳新的 BoardState（immutable）
  BoardState setStone(int row, int col, StoneColor color) {
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
      throw RangeError(
          'Position ($row, $col) out of bounds for $rows x $cols board');
    }
    final newGrid = List.generate(
      rows,
      (r) => List<StoneColor>.from(grid[r]),
    );
    newGrid[row][col] = color;
    return BoardState(
      rows: rows,
      cols: cols,
      grid: newGrid,
      nextPlayer: nextPlayer,
      komi: komi,
      isPartial: isPartial,
      realEdges: realEdges,
    );
  }

  /// 計算棋盤上黑子數量
  int get blackCount {
    int count = 0;
    for (final row in grid) {
      for (final stone in row) {
        if (stone == StoneColor.black) count++;
      }
    }
    return count;
  }

  /// 計算棋盤上白子數量
  int get whiteCount {
    int count = 0;
    for (final row in grid) {
      for (final stone in row) {
        if (stone == StoneColor.white) count++;
      }
    }
    return count;
  }

  /// 檢查棋盤是否為空
  bool get isEmpty => blackCount == 0 && whiteCount == 0;

  /// 將棋盤轉換為 flat list（row-major order）
  List<StoneColor> toFlatList() {
    return grid.expand((row) => row).toList();
  }

  @override
  String toString() {
    final sb = StringBuffer();
    final label = isPartial ? '${rows}x$cols (局部)' : '${rows}x$cols';
    sb.writeln('BoardState $label, next: $nextPlayer');
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        switch (grid[r][c]) {
          case StoneColor.empty:
            sb.write('.');
          case StoneColor.black:
            sb.write('X');
          case StoneColor.white:
            sb.write('O');
        }
      }
      sb.writeln();
    }
    return sb.toString();
  }
}
