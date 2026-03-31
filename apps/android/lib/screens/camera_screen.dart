import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'result_screen.dart';

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _initializing = true;
  String? _error;
  int _gridSize = 19;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      _controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _error = '找不到可用的相機';
          _initializing = false;
        });
        return;
      }

      final controller = CameraController(
        cameras.first,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) return;

      setState(() {
        _controller = controller;
        _initializing = false;
      });
    } catch (e) {
      setState(() {
        _error = '相機初始化失敗: $e';
        _initializing = false;
      });
    }
  }

  Future<void> _capture() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    try {
      final xFile = await _controller!.takePicture();
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(imagePath: xFile.path),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('拍照失敗: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('對準棋盤'),
        actions: [
          PopupMenuButton<int>(
            icon: const Icon(Icons.grid_on),
            tooltip: '格線大小',
            onSelected: (size) => setState(() => _gridSize = size),
            itemBuilder: (_) => [9, 13, 19].map((s) =>
              PopupMenuItem(value: s, child: Text('${s}x$s')),
            ).toList(),
          ),
        ],
      ),
      body: _initializing
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(child: Text(_error!, style: const TextStyle(color: Colors.white)))
              : _buildPreview(),
    );
  }

  Widget _buildPreview() {
    return Column(
      children: [
        Expanded(
          child: ClipRect(
            child: Stack(
              fit: StackFit.expand,
              children: [
                // 相機預覽
                Center(child: CameraPreview(_controller!)),
                // 格線 overlay
                Center(
                  child: AspectRatio(
                    aspectRatio: 1,
                    child: CustomPaint(
                      painter: _GridOverlayPainter(gridSize: _gridSize),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        // 拍照按鈕
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text('${_gridSize}x$_gridSize',
                    style: const TextStyle(color: Colors.white54, fontSize: 14)),
                const SizedBox(width: 32),
                GestureDetector(
                  onTap: _capture,
                  child: Container(
                    width: 72,
                    height: 72,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white, width: 4),
                    ),
                    child: Container(
                      margin: const EdgeInsets.all(4),
                      decoration: const BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 32 + 30), // balance
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// 棋盤格線 overlay — 半透明格線幫助對齊
class _GridOverlayPainter extends CustomPainter {
  final int gridSize;

  _GridOverlayPainter({required this.gridSize});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green.withValues(alpha: 0.4)
      ..strokeWidth = 0.8;

    final padding = size.width * 0.04;
    final cellSize = (size.width - padding * 2) / (gridSize - 1);

    for (int i = 0; i < gridSize; i++) {
      final pos = padding + i * cellSize;
      // 垂直線
      canvas.drawLine(Offset(pos, padding), Offset(pos, size.height - padding), paint);
      // 水平線
      canvas.drawLine(Offset(padding, pos), Offset(size.width - padding, pos), paint);
    }

    // 外框加粗
    final borderPaint = Paint()
      ..color = Colors.green.withValues(alpha: 0.6)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    canvas.drawRect(
      Rect.fromLTRB(padding, padding, size.width - padding, size.height - padding),
      borderPaint,
    );
  }

  @override
  bool shouldRepaint(_GridOverlayPainter oldDelegate) =>
      oldDelegate.gridSize != gridSize;
}
