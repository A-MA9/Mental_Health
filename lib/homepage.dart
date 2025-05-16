import 'package:flutter/material.dart';
import 'package:intl/intl.dart'; // for date formatting
import 'package:supabase_flutter/supabase_flutter.dart';
import 'main.dart'; // Import to access the supabase client
import 'signin.dart';
import 'account.dart'; // Import to access the AccountPage
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'transitions.dart'; // Import for custom transitions
import 'cameraaudio.dart'; // Import to access the CameraAudioPage
import 'package:flutter/services.dart';
import 'model_selection.dart'; // Import to access the ModelSelectionPage
import 'stats_page.dart'; // Import to access the StatsPage

class HomePage extends StatefulWidget {
  final VoidCallback onOpenSettings;
  const HomePage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String _userName = "User";
  String _userEmail = "";
  String? _avatarUrl;
  bool _isLoading = true;
  int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    _checkAuthStatus();
    _loadUserProfile();
  }

  Future<void> _checkAuthStatus() async {
    final session = supabase.auth.currentSession;
    if (session == null) {
      debugPrint('User is not authenticated, redirecting to sign in');
      _navigateToSignIn();
    } else {
      supabase.auth.onAuthStateChange.listen((data) {
        final AuthChangeEvent event = data.event;
        if (event == AuthChangeEvent.signedOut) {
          debugPrint('Auth state changed: User signed out');
          _navigateToSignIn();
        }
      });
    }
  }

  void _navigateToSignIn() {
    Navigator.of(context).pushAndRemoveUntil(
      fadeTransition(
        SignInPage(
          onSignInSuccess: () {},
        ),
      ),
      (route) => false,
    );
  }

  Future<void> _loadUserProfile() async {
    setState(() => _isLoading = true);
    try {
      final User? user = supabase.auth.currentUser;

      if (user != null) {
        final profileData =
            await supabase.from('profiles').select().eq('id', user.id).single();

        setState(() {
          _userName = profileData['name'] ?? 'User';
          _userEmail = user.email ?? '';
          _avatarUrl = profileData['avatar_url'] as String?;
        });
      }
    } catch (e) {
      debugPrint('Error loading profile: $e');
      if (mounted) {
        setState(() {
          _userName = 'User';
          _userEmail = 'Email not found';
        });
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Future<void> _signOut() async {
    try {
      await supabase.auth.signOut();
      if (mounted) {
        _navigateToSignIn();
      }
    } catch (e) {
      debugPrint('Error signing out: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Error signing out. Please try again.'),
            duration: Duration(seconds: 2),
          ),
        );
      }
    }
  }

  void _onNavTap(int index) {
    setState(() {
      _selectedIndex = index;
    });
    if (index == 0) {
      // Already on Home
    } else if (index == 1) {
      Navigator.pushReplacement(
        context,
        fadeTransition(
          ModeSelectionPage(onOpenSettings: widget.onOpenSettings),
        ),
      );
    } else if (index == 2) {
      Navigator.pushReplacement(
        context,
        fadeTransition(
          StatsPage(onOpenSettings: widget.onOpenSettings),
        ),
      );
    } else if (index == 3) {
      Navigator.pushReplacement(
        context,
        fadeTransition(
          AccountPage(onOpenSettings: widget.onOpenSettings),
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF1B2B1A) : null;
    final cardColor = isDark ? const Color(0xFF223D1B) : Colors.white;
    final textColor = isDark ? Colors.white : Colors.black;
    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: isDark ? const Color(0xFF223D1B) : Colors.lightGreen[100],
        statusBarIconBrightness: isDark ? Brightness.light : Brightness.dark,
      ),
    );
    String formattedDate = DateFormat('dd MMM yyyy').format(DateTime.now());

    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            ClipRRect(
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(20),
                bottomRight: Radius.circular(20),
              ),
              child: Container(
                color: isDark ? const Color(0xFF223D1B) : Colors.lightGreen[100],
                padding: const EdgeInsets.fromLTRB(16, 10, 16, 16),
                child: Column(
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Row(
                          children: [
                            Icon(Icons.today, color: isDark ? Colors.white : Colors.brown),
                            const SizedBox(width: 8),
                            Text(formattedDate, style: TextStyle(fontSize: 14, color: textColor)),
                          ],
                        ),
                        Row(
                          children: [
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: isDark ? Colors.white10 : Colors.white,
                                shape: BoxShape.circle,
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.grey.withOpacity(0.3),
                                    spreadRadius: 1,
                                    blurRadius: 4,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: Icon(
                                Icons.notifications_none,
                                size: 24,
                                color: isDark ? Colors.white : Colors.black87,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        CircleAvatar(
                          radius: 28,
                          backgroundColor: Colors.brown[700],
                          backgroundImage: (_avatarUrl != null && _avatarUrl!.isNotEmpty)
                              ? NetworkImage(_avatarUrl!)
                              : null,
                          child: (_avatarUrl == null || _avatarUrl!.isEmpty)
                              ? Text(
                            _isLoading
                                ? '?'
                                : _userName.isNotEmpty
                                ? _userName[0].toUpperCase()
                                : 'U',
                                  style: const TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Colors.white,
                                  ),
                                )
                              : null,
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _isLoading ? 'Loading...' : _userName,
                                style: TextStyle(
                                  fontSize: 20,
                                  fontWeight: FontWeight.bold,
                                  color: textColor,
                                ),
                              ),
                              const SizedBox(height: 4),
                              Text(
                                _isLoading ? '' : _userEmail,
                                style: TextStyle(
                                  fontSize: 14,
                                  color: isDark ? Colors.white70 : Colors.grey[700],
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 12),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  fadeTransition(
                    ModeSelectionPage(onOpenSettings: widget.onOpenSettings),
                  ),
                );
              },
              child: Stack(
                children: [
                  Container(
                    width: double.infinity,
                    height: 270,
                    margin: const EdgeInsets.symmetric(horizontal: 12),
                    decoration: BoxDecoration(
                      color: isDark ? const Color(0xFF223D1B) : const Color(0xFFC1E1C1),
                      borderRadius: BorderRadius.circular(36),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black12,
                          blurRadius: 24,
                          offset: Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Row(
                      children: [
                        // Large bot image on the left, overlapping the card edge
                        Container(
                          height: 270,
                          width: 160,
                          alignment: Alignment.centerLeft,
                          child: OverflowBox(
                            maxHeight: 260,
                            maxWidth: 200,
                            alignment: Alignment.centerLeft,
                            child: Image.network(
                              'https://cqfwjwrhhcctazigevno.supabase.co/storage/v1/object/public/assets//3D-Hello-GIF-by-L3S-Research-C-unscreen.gif',
                              height: 200,
                              width: 200,
                              fit: BoxFit.contain,
                            ),
                          ),
                        ),
                        Expanded(
                          child: Padding(
                            padding: const EdgeInsets.fromLTRB(24, 36, 24, 24),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Text(
                                  "I'm here to listen",
                                  style: TextStyle(
                                    fontSize: 32,
                                    fontWeight: FontWeight.w900,
                                    color: isDark ? Colors.white : Colors.black87,
                                    letterSpacing: 1.1,
                                    height: 1.1,
                                    shadows: [
                                      Shadow(
                                        color: isDark ? Colors.black26 : Colors.white70,
                                        blurRadius: 6,
                                        offset: Offset(0, 2),
                                      ),
                                    ],
                                  ),
                                ),
                                const SizedBox(height: 10),
                                Text(
                                  "Your AI companion",
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.w500,
                                    color: isDark ? Colors.blue[200] : Colors.blue[700],
                                    letterSpacing: 0.5,
                                  ),
                                ),
                                const Spacer(),
                                Center(
                                  child: Container(
                                    decoration: BoxDecoration(
                                      color: isDark ? Colors.blueGrey[700] : Colors.white,
                                      borderRadius: BorderRadius.circular(24),
                                      boxShadow: [
                                        BoxShadow(
                                          color: Colors.black12,
                                          blurRadius: 8,
                                          offset: Offset(0, 2),
                                        ),
                                      ],
                                    ),
                                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                                    child: Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        Icon(Icons.chat_bubble_outline, color: isDark ? Colors.white : Colors.blueAccent, size: 28),
                                        const SizedBox(width: 12),
                                        Text(
                                          "Chat",
                                          style: TextStyle(
                                            color: isDark ? Colors.white : Colors.blueAccent,
                                            fontWeight: FontWeight.w700,
                                            fontSize: 20,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        ],
                      ),
                  ),
                ],
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
    );
  }
}
