import 'package:flutter/material.dart';
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'homepage.dart';
import 'model_selection.dart';
import 'account.dart';

class StatsPage extends StatefulWidget {
  final VoidCallback onOpenSettings;
  const StatsPage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  State<StatsPage> createState() => _StatsPageState();
}

class _StatsPageState extends State<StatsPage> {
  int _selectedIndex = 2; // Stats tab

  void _onNavTap(int index) {
    setState(() {
      _selectedIndex = index;
    });
    if (index == 0) {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => HomePage(onOpenSettings: widget.onOpenSettings),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var tween = Tween(begin: 0.0, end: 1.0);
            var fadeAnimation = animation.drive(tween);
            return FadeTransition(opacity: fadeAnimation, child: child);
          },
        ),
      );
    } else if (index == 1) {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => ModeSelectionPage(onOpenSettings: widget.onOpenSettings),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var tween = Tween(begin: 0.0, end: 1.0);
            var fadeAnimation = animation.drive(tween);
            return FadeTransition(opacity: fadeAnimation, child: child);
          },
        ),
      );
    } else if (index == 2) {
      // Already on Stats
    } else if (index == 3) {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => AccountPage(onOpenSettings: widget.onOpenSettings),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            var tween = Tween(begin: 0.0, end: 1.0);
            var fadeAnimation = animation.drive(tween);
            return FadeTransition(opacity: fadeAnimation, child: child);
          },
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Stats'),
        centerTitle: true,
        actions: [
          IconButton(
            icon: AnimatedSwitcher(
              duration: const Duration(milliseconds: 400),
              transitionBuilder: (child, animation) => RotationTransition(turns: animation, child: child),
              child: isDark
                  ? const Icon(Icons.dark_mode, key: ValueKey('moon'), color: Colors.white)
                  : const Icon(Icons.wb_sunny, key: ValueKey('sun'), color: Colors.black),
            ),
            onPressed: widget.onOpenSettings,
          ),
        ],
      ),
      body: const Center(
        child: Text('Stats Page Coming Soon!', style: TextStyle(fontSize: 24)),
      ),
      bottomNavigationBar: CurvedNavigationBar(
        index: _selectedIndex,
        height: 60.0,
        color: isDark ? const Color(0xFF223D1B) : Colors.lightGreen,
        backgroundColor: Colors.transparent,
        animationDuration: const Duration(milliseconds: 300),
        items: const <Widget>[
          Icon(Icons.home, size: 30, color: Colors.white),
          Icon(Icons.chat, size: 30, color: Colors.white),
          Icon(Icons.stacked_bar_chart, size: 30, color: Colors.white),
          Icon(Icons.person, size: 30, color: Colors.white),
        ],
        onTap: _onNavTap,
      ),
      backgroundColor: isDark ? const Color(0xFF1B2B1A) : Colors.white,
    );
  }
} 