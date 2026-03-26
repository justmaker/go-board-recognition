import 'dart:io';
import 'package:flutter/material.dart';
import 'package:go_board_core/go_board_core.dart';
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
      );
      final result = await recognition.recognizeFromImage(widget.imagePath);
      setState(() {
        _result = result;
        _loading = false;
      });
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
          if (_result != null)
            IconButton(
              icon: Icon(_showDebugOverlay ? Icons.bug_report : Icons.bug_report_outlined),
              onPressed: () => setState(() => _showDebugOverlay = !_showDebugOverlay),
              tooltip: 'Debug overlay',
            ),
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

  Widget _buildResult() {
    final result = _result!;
    final board = result.boardState;
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
                '辨識結果: ${board.boardSize}x${board.boardSize}  黑=${board.blackCount}  白=${board.whiteCount}',
                style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            const SizedBox(height: 8),
            AspectRatio(
              aspectRatio: 1,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: CustomPaint(
                  painter: DebugBoardPainter(boardState: board),
                ),
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
