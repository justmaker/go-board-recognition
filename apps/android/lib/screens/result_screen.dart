import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_board_core/go_board_core.dart';
import 'package:share_plus/share_plus.dart';
import '../services/history_service.dart';
import '../widgets/debug_board_painter.dart';

class ResultScreen extends StatefulWidget {
  final String imagePath;

  const ResultScreen({super.key, required this.imagePath});

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  RecognitionResult? _result;
  String? _error;
  bool _loading = true;
  bool _showDebugOverlay = true;
  bool _editMode = false;
  bool _useCNN = true; // Default to CNN mode
  BoardState? _editableBoard;
  final List<String> _logs = [];

  @override
  void initState() {
    super.initState();
    _recognize();
  }

  Future<void> _recognize() async {
    setState(() {
      _loading = true;
      _error = null;
      _logs.clear();
    });

    try {
      final recognition = BoardRecognition(
        onLog: (msg) => setState(() => _logs.add(msg)),
        keepWarpedImage: true,
        useCNN: _useCNN,
      );
      final result = await recognition.recognizeFromImage(widget.imagePath);
      setState(() {
        _result = result;
        _loading = false;
      });
      // 自動儲存到歷史紀錄
      final board = result.boardState;
      await HistoryService.save(ScanRecord(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.now(),
        imagePath: widget.imagePath,
        rows: board.rows,
        cols: board.cols,
        blackCount: board.blackCount,
        whiteCount: board.whiteCount,
      ));
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('辨識結果'),
        actions: [
          if (_result != null) ...[
            IconButton(
              icon: Icon(_useCNN ? Icons.psychology : Icons.auto_fix_high),
              onPressed: () {
                setState(() {
                  _useCNN = !_useCNN;
                });
                _recognize(); // Re-run with new mode
              },
              tooltip: _useCNN ? 'CNN 模式 (切換到 CV)' : 'CV 模式 (切換到 CNN)',
            ),
            IconButton(
              icon: Icon(_editMode ? Icons.edit_off : Icons.edit),
              onPressed: () => setState(() {
                _editMode = !_editMode;
                if (_editMode && _editableBoard == null) {
                  _editableBoard = _result!.boardState;
                }
              }),
              tooltip: '手動修正',
            ),
            IconButton(
              icon: const Icon(Icons.share),
              onPressed: _exportSgf,
              tooltip: '匯出 SGF',
            ),
            IconButton(
              icon: Icon(_showDebugOverlay ? Icons.bug_report : Icons.bug_report_outlined),
              onPressed: () => setState(() => _showDebugOverlay = !_showDebugOverlay),
              tooltip: 'Debug overlay',
            ),
          ],
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? _buildError()
              : _buildResult(),
    );
  }

  Widget _buildError() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(_error!, textAlign: TextAlign.center),
            const SizedBox(height: 24),
            FilledButton(onPressed: _recognize, child: const Text('重試')),
          ],
        ),
      ),
    );
  }

  Future<void> _exportSgf() async {
    final board = _editMode ? (_editableBoard ?? _result!.boardState) : _result!.boardState;
    final sgf = SgfExport.toSgf(board, comment: '由 BoardScanner 辨識匯出');
    final tempFile = File('${Directory.systemTemp.path}/board_scan.sgf');
    await tempFile.writeAsString(sgf);
    await Share.shareXFiles([XFile(tempFile.path)]);
  }

  void _handleBoardTap(TapDownDetails details, Size boardSize, BoardState board) {
    final padding = boardSize.shortestSide * 0.04;
    final cellW = (boardSize.width - padding * 2) / (board.cols - 1);
    final cellH = (boardSize.height - padding * 2) / (board.rows - 1);

    final col = ((details.localPosition.dx - padding) / cellW).round();
    final row = ((details.localPosition.dy - padding) / cellH).round();

    if (row < 0 || row >= board.rows || col < 0 || col >= board.cols) return;

    final current = board.getStone(row, col);
    final next = switch (current) {
      StoneColor.empty => StoneColor.black,
      StoneColor.black => StoneColor.white,
      StoneColor.white => StoneColor.empty,
    };

    setState(() {
      _editableBoard = board.setStone(row, col, next);
    });
  }

  Widget _buildResult() {
    final result = _result!;
    final board = _editMode ? (_editableBoard ?? result.boardState) : result.boardState;
    final debug = result.debugInfo;

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // 原始影像
          AspectRatio(
            aspectRatio: 1,
            child: Image.file(File(widget.imagePath), fit: BoxFit.contain),
          ),

          // 辨識後的棋盤 (debug overlay)
          if (_showDebugOverlay) ...[
            const Divider(),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Text(
                '辨識結果: ${board.rows}x${board.cols}'
                '${board.isPartial ? ' (局部)' : ''}'
                '  黑=${board.blackCount}  白=${board.whiteCount}',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 8),
            AspectRatio(
              aspectRatio: board.cols / board.rows,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final boardSize = Size(constraints.maxWidth, constraints.maxWidth * board.rows / board.cols);
                    return GestureDetector(
                      onTapDown: _editMode
                          ? (details) => _handleBoardTap(details, boardSize, board)
                          : null,
                      child: CustomPaint(
                        painter: DebugBoardPainter(boardState: board),
                      ),
                    );
                  },
                ),
              ),
            ),
            if (_editMode)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16),
                child: Text(
                  '點擊交叉點修正：空 → 黑 → 白 → 空',
                  style: TextStyle(color: Colors.orange[800], fontSize: 13),
                ),
              ),
          ],

          // Debug info
          if (_showDebugOverlay) ...[
            const Divider(),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Debug Info',
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  Text(debug.toString(),
                      style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
                  if (_logs.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    const Text('Logs',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    ..._logs.map((log) => Text(log,
                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11))),
                  ],
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }
}
