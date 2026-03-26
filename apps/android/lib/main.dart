import 'package:flutter/material.dart';
import 'screens/home_screen.dart';

void main() {
  runApp(const BoardScannerApp());
}

class BoardScannerApp extends StatelessWidget {
  const BoardScannerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Board Scanner',
      theme: ThemeData(
        colorSchemeSeed: Colors.brown,
        useMaterial3: true,
        brightness: Brightness.dark,
      ),
      home: const HomeScreen(),
    );
  }
}
