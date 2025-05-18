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

class _ModeSelectionPageState extends State<ModeSelectionPage> with SingleTickerProviderStateMixin {
  int _selectedIndex = 1; // assuming this is the "chat" tab
  bool _gifArrived = false;

  @override
  void initState() {
    super.initState();
    // Trigger the animation after the first frame
    WidgetsBinding.instance.addPostFrameCallback((_) {
      setState(() {
        _gifArrived = true;
      });
    });
  }

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
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final bgColor = theme.scaffoldBackgroundColor;
    final cardColor = theme.cardColor;
    final textColor = theme.textTheme.bodyLarge?.color ?? (isDark ? Colors.white : Colors.black);
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video Interaction Mode'),
        centerTitle: true,
        backgroundColor: theme.appBarTheme.backgroundColor,
        iconTheme: theme.appBarTheme.iconTheme,
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
      body: Stack(
        children: [
          // Revert to original solid background color from theme
          Container(color: bgColor),
          // Top center GIF animation
          AnimatedAlign(
            alignment: _gifArrived ? Alignment.topCenter : Alignment(-1.2, -1.2),
            duration: const Duration(milliseconds: 1800),
            curve: Curves.easeOutBack,
            child: AnimatedOpacity(
              opacity: _gifArrived ? 1.0 : 0.0,
              duration: const Duration(milliseconds: 1400),
              child: Padding(
                padding: const EdgeInsets.only(top: 0.0),
                child: SizedBox(
                  width: 300,
                  height: 300,
                  child: Image.network(
                    'https://cqfwjwrhhcctazigevno.supabase.co/storage/v1/object/public/assets//1-IaGqJTdADQIy79k-EI-ZUw-unscreen.gif',
                    fit: BoxFit.contain,
                  ),
                ),
              ),
            ),
          ),
          // Main content
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Stack(
              children: [
                // Centered Vi Talk button
                Center(
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.push(
                        context,
                        fadeTransition(
                          AudioUploaderWidget()
                        ),
                      );
                    },
                    icon: Icon(Icons.videocam, color: Colors.white, size: 32),
                    label: const Text(
                      'Vi Talk',
                      style: TextStyle(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                    style: ElevatedButton.styleFrom(
                      minimumSize: const Size(double.infinity, 70),
                      backgroundColor: Colors.deepPurple,
                      textStyle: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(40),
                      ),
                      elevation: 6,
                    ),
                  ),
                ),
                // Glowing info box for upcoming features at the bottom
                Align(
                  alignment: Alignment.bottomCenter,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 32.0),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 20),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.95),
                        borderRadius: BorderRadius.circular(24),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.deepPurple.withOpacity(0.4),
                            blurRadius: 32,
                            spreadRadius: 2,
                            offset: const Offset(0, 0),
                          ),
                        ],
                      ),
                      child: const Text(
                        'More features will be added in coming update.\nStay tuned!',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w600,
                          color: Colors.black,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
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

// 3D Texture Painter
class _TexturePainter extends CustomPainter {
  final bool isDark;
  _TexturePainter({required this.isDark});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = isDark ? Colors.black.withOpacity(0.05) : Colors.blueGrey.withOpacity(0.07)
      ..strokeWidth = 2;
    // Draw diagonal lines for a subtle 3D effect
    for (double i = -size.height; i < size.width; i += 24) {
      canvas.drawLine(
        Offset(i, 0),
        Offset(i + size.height, size.height),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
