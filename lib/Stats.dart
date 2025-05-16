import 'package:flutter/material.dart';

class StatsPage extends StatelessWidget {
  final VoidCallback onOpenSettings;
  const StatsPage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Stats'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: onOpenSettings,
          ),
        ],
      ),
      body: const Center(
        child: Text('Stats Page Coming Soon!', style: TextStyle(fontSize: 24)),
      ),
    );
  }
}
