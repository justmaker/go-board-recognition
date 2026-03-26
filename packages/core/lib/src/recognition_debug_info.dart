/// 辨識管線除錯資訊
class RecognitionDebugInfo {
  String boardDetectionMethod = '';
  int detectedBoardSize = 0;
  int hLineCount = 0;
  int vLineCount = 0;
  double hSpacing = 0;
  double vSpacing = 0;
  List<double> clusterCenters = [];
  double thresholdBlackBoard = 0;
  double thresholdBoardWhite = 0;
  double satLimitBlack = 0;
  double satLimitWhite = 0;
  int blackCount = 0;
  int whiteCount = 0;
  int emptyCount = 0;
  double vMin = 0;
  double vMax = 0;

  @override
  String toString() {
    return '''
=== 棋盤辨識除錯 ===
板面偵測: $boardDetectionMethod
格線: H=$hLineCount, V=$vLineCount → ${detectedBoardSize}x$detectedBoardSize
間距: H=${hSpacing.toStringAsFixed(1)}, V=${vSpacing.toStringAsFixed(1)}
V 範圍: ${vMin.toStringAsFixed(1)} ~ ${vMax.toStringAsFixed(1)}
聚類中心: ${clusterCenters.map((c) => c.toStringAsFixed(1)).join(', ')}
閾值: B<${thresholdBlackBoard.toStringAsFixed(1)} S<${satLimitBlack.toStringAsFixed(1)}, W>${thresholdBoardWhite.toStringAsFixed(1)} S<${satLimitWhite.toStringAsFixed(1)}
結果: 黑=$blackCount, 白=$whiteCount, 空=$emptyCount
====================''';
  }
}
