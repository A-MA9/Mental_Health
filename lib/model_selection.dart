import 'package:flutter/material.dart';
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'transitions.dart';
import 'account.dart';
import 'homepage.dart';
import 'cameraaudio.dart';
import 'stats_page.dart';
import 'resultpage.dart';

class ModeSelectionPage extends StatefulWidget {
  final VoidCallback onOpenSettings;
  const ModeSelectionPage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  State<ModeSelectionPage> createState() => _ModeSelectionPageState();
}

class _ModeSelectionPageState extends State<ModeSelectionPage> {
  int _selectedIndex = 1; // assuming this is the "chat" tab

  void _onNavTap(int index) {
    if (_selectedIndex == index) return; // Prevent unnecessary navigation
    Widget targetPage;
    int? targetIndex;
    if (index == 0) {
      targetPage = HomePage(onOpenSettings: widget.onOpenSettings);
    } else if (index == 1) {
      // Already on ModeSelectionPage
      return;
    } else if (index == 2) {
    
      
      targetPage = StatsPage(onOpenSettings: widget.onOpenSettings);

    } else if (index == 3) {
      targetPage = AccountPage(onOpenSettings: widget.onOpenSettings);
      targetIndex = 3;
    } else {
      return;
    }
    Navigator.of(context).pushAndRemoveUntil(
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => targetIndex != null
            ? AccountPage(onOpenSettings: widget.onOpenSettings)
            : targetPage,
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          var tween = Tween(begin: 0.0, end: 1.0);
          var fadeAnimation = animation.drive(tween);
          return FadeTransition(opacity: fadeAnimation, child: child);
        },
        settings: targetIndex != null ? RouteSettings(arguments: targetIndex) : null,
      ),
      (route) => false,
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final args = ModalRoute.of(context)?.settings.arguments;
    if (args is int && args != _selectedIndex) {
      setState(() {
        _selectedIndex = args;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF1B2B1A) : null;
    final cardColor = isDark ? const Color(0xFF223D1B) : Colors.white;
    final textColor = isDark ? Colors.white : Colors.black;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Choose Interaction Mode'),
        centerTitle: true,
        backgroundColor: isDark ? const Color(0xFF223D1B) : null,
        iconTheme: IconThemeData(color: isDark ? Colors.white : Colors.black),
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
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                // Navigate to ChatScreen
              },
              icon: Icon(Icons.chat, color: isDark ? Colors.white : Colors.black),
              label: Text('Chat with Me', style: TextStyle(color: textColor)),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 50),
                textStyle: const TextStyle(fontSize: 18),
                backgroundColor: cardColor,
                foregroundColor: textColor,
              ),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () {
                Navigator.push(
                  context,
                  fadeTransition(
                    AudioUploaderWidget()
                  ),
                );
                // Navigate to VideoChatScreen
              },
              icon: Icon(Icons.videocam, color: Colors.white),
              label: const Text(
                'Vi Talk',
                style: TextStyle(color: Colors.white),
              ),
              style: ElevatedButton.styleFrom(
                minimumSize: const Size(double.infinity, 50),
                backgroundColor: Colors.deepPurple,
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
          ],
        ),
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
      backgroundColor: bgColor,
    );
  }
}
