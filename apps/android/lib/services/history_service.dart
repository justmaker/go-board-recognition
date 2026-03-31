import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// 辨識紀錄
class ScanRecord {
  final String id;
  final DateTime timestamp;
  final String imagePath;
  final int rows;
  final int cols;
  final int blackCount;
  final int whiteCount;

  ScanRecord({
    required this.id,
    required this.timestamp,
    required this.imagePath,
    required this.rows,
    required this.cols,
    required this.blackCount,
    required this.whiteCount,
  });

  Map<String, dynamic> toJson() => {
    'id': id,
    'timestamp': timestamp.toIso8601String(),
    'imagePath': imagePath,
    'rows': rows,
    'cols': cols,
    'blackCount': blackCount,
    'whiteCount': whiteCount,
  };

  factory ScanRecord.fromJson(Map<String, dynamic> json) => ScanRecord(
    id: json['id'] as String,
    timestamp: DateTime.parse(json['timestamp'] as String),
    imagePath: json['imagePath'] as String,
    rows: json['rows'] as int,
    cols: json['cols'] as int,
    blackCount: json['blackCount'] as int,
    whiteCount: json['whiteCount'] as int,
  );
}

/// 管理辨識歷史紀錄的服務
class HistoryService {
  static const _fileName = 'scan_history.json';

  static Future<File> get _file async {
    final dir = await getApplicationDocumentsDirectory();
    return File('${dir.path}/$_fileName');
  }

  /// 載入所有紀錄（新的在前）
  static Future<List<ScanRecord>> loadAll() async {
    final file = await _file;
    if (!file.existsSync()) return [];

    final json = jsonDecode(await file.readAsString()) as List;
    final records = json.map((e) => ScanRecord.fromJson(e as Map<String, dynamic>)).toList();
    records.sort((a, b) => b.timestamp.compareTo(a.timestamp));
    return records;
  }

  /// 新增一筆紀錄，同時複製圖片到 app 目錄
  static Future<void> save(ScanRecord record) async {
    // 複製原始圖片到 app 目錄
    final dir = await getApplicationDocumentsDirectory();
    final imagesDir = Directory('${dir.path}/scan_images');
    if (!imagesDir.existsSync()) imagesDir.createSync();

    final srcFile = File(record.imagePath);
    final ext = record.imagePath.split('.').last;
    final destPath = '${imagesDir.path}/${record.id}.$ext';
    if (srcFile.existsSync()) {
      await srcFile.copy(destPath);
    }

    final saved = ScanRecord(
      id: record.id,
      timestamp: record.timestamp,
      imagePath: destPath,
      rows: record.rows,
      cols: record.cols,
      blackCount: record.blackCount,
      whiteCount: record.whiteCount,
    );

    final records = await loadAll();
    records.insert(0, saved);

    final file = await _file;
    await file.writeAsString(jsonEncode(records.map((r) => r.toJson()).toList()));
  }

  /// 刪除一筆紀錄
  static Future<void> delete(String id) async {
    final records = await loadAll();
    final record = records.where((r) => r.id == id).firstOrNull;
    if (record != null) {
      final imgFile = File(record.imagePath);
      if (imgFile.existsSync()) imgFile.deleteSync();
    }

    records.removeWhere((r) => r.id == id);
    final file = await _file;
    await file.writeAsString(jsonEncode(records.map((r) => r.toJson()).toList()));
  }
}
