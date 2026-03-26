import 'dart:io';
import 'package:flutter/material.dart';
import '../services/history_service.dart';
import 'result_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  List<ScanRecord>? _records;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final records = await HistoryService.loadAll();
    setState(() => _records = records);
  }

  Future<void> _delete(ScanRecord record) async {
    await HistoryService.delete(record.id);
    _load();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('辨識紀錄')),
      body: _records == null
          ? const Center(child: CircularProgressIndicator())
          : _records!.isEmpty
              ? const Center(child: Text('尚無紀錄'))
              : ListView.builder(
                  itemCount: _records!.length,
                  itemBuilder: (context, index) {
                    final record = _records![index];
                    final imageFile = File(record.imagePath);
                    final timeStr = '${record.timestamp.month}/${record.timestamp.day} '
                        '${record.timestamp.hour.toString().padLeft(2, '0')}:'
                        '${record.timestamp.minute.toString().padLeft(2, '0')}';

                    return Dismissible(
                      key: Key(record.id),
                      direction: DismissDirection.endToStart,
                      background: Container(
                        color: Colors.red,
                        alignment: Alignment.centerRight,
                        padding: const EdgeInsets.only(right: 16),
                        child: const Icon(Icons.delete, color: Colors.white),
                      ),
                      onDismissed: (_) => _delete(record),
                      child: ListTile(
                        leading: SizedBox(
                          width: 56,
                          height: 56,
                          child: imageFile.existsSync()
                              ? Image.file(imageFile, fit: BoxFit.cover)
                              : const Icon(Icons.broken_image),
                        ),
                        title: Text('${record.rows}x${record.cols}  黑${record.blackCount} 白${record.whiteCount}'),
                        subtitle: Text(timeStr),
                        onTap: () {
                          if (imageFile.existsSync()) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (_) => ResultScreen(imagePath: record.imagePath),
                              ),
                            );
                          }
                        },
                      ),
                    );
                  },
                ),
    );
  }
}
