import 'package:flutter/material.dart';
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'main.dart';
import 'signin.dart';
import 'homepage.dart';
import 'transitions.dart';
import 'package:flutter/services.dart';
import 'edit_profile.dart';
import 'package:http/http.dart' as http;
import 'faq_screen.dart';
import 'settings_screen.dart';
import 'stats_page.dart';
import 'model_selection.dart';

class AccountPage extends StatefulWidget {
  final VoidCallback onOpenSettings;
  const AccountPage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  State<AccountPage> createState() => _AccountPageState();
}

class _AccountPageState extends State<AccountPage> {
  String _userName = "User";
  String _userEmail = "";
  DateTime? _userDob;
  String? _avatarUrl;
  bool _isLoading = true;
  bool _isDeleting = false;

  int _selectedIndex = 3; // Account Index - 3

  @override
  void initState() {
    super.initState();
    // Set the status bar color to light green
    SystemChrome.setSystemUIOverlayStyle(
      SystemUiOverlayStyle(
        statusBarColor: Colors.lightGreen[100],
        statusBarIconBrightness: Brightness.dark,
      ),
    );
    _loadUserProfile();
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

  Future<void> _loadUserProfile() async {
    setState(() => _isLoading = true);

    try {
      final user = supabase.auth.currentUser;
      if (user != null) {
        final profile =
            await supabase.from('profiles').select().eq('id', user.id).single();

        setState(() {
          _userName = profile['name'] ?? 'User';
          _userEmail = user.email ?? '';
          _userDob = profile['date_of_birth'] != null && profile['date_of_birth'] != ''
              ? DateTime.tryParse(profile['date_of_birth'])
              : null;
          _avatarUrl = profile['avatar_url'] as String?;
        });
      }
    } catch (e) {
      debugPrint('Error loading profile: $e');
      final user = supabase.auth.currentUser;
      if (user != null) {
        setState(() {
          _userName = 'User';
          _userEmail = user.email ?? '';
          _userDob = null;
          _avatarUrl = null;
        });
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  Future<void> _signOut() async {
    try {
      await supabase.auth.signOut();
      if (mounted) {
        Navigator.of(context).pushAndRemoveUntil(
          fadeTransition(
            SignInPage(
              onSignInSuccess: () {},
            ),
          ),
          (route) => false,
        );
      }
    } catch (e) {
      debugPrint('Sign out error: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(const SnackBar(content: Text('Error signing out.')));
      }
    }
  }

  Future<void> _navigateToEditProfile() async {
    final result = await Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => EditProfilePage(
          initialName: _userName,
          initialEmail: _userEmail,
          initialDob: _userDob,
        ),
      ),
    );
    if (result == true) {
      // Refresh profile after editing
      _loadUserProfile();
    }
  }

  Future<void> _deleteAccount() async {
    final user = supabase.auth.currentUser;
    if (user == null) return;
    setState(() => _isDeleting = true);
    try {
      final response = await http.post(
        Uri.parse('https://cqfwjwrhhcctazigevno.supabase.co/functions/v1/delete_user_account'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${supabase.auth.currentSession?.accessToken ?? ''}',
        },
        body: '{"user_id": "${user.id}"}',
      );
      if (response.statusCode == 200) {
        await supabase.auth.signOut();
        if (mounted) {
          Navigator.of(context).pushAndRemoveUntil(
            fadeTransition(
              SignInPage(
                onSignInSuccess: () {},
              ),
            ),
            (route) => false,
          );
        }
      } else {
        throw Exception('Failed to delete account');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error deleting account: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isDeleting = false);
    }
  }

  void _showDeleteAccountDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Account'),
        content: const Text('Do you really want to delete your account? This action cannot be undone.'),
        actions: [
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.grey[200],
                    foregroundColor: Colors.black,
                    elevation: 4,
                    shadowColor: Colors.grey[400],
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                      side: const BorderSide(color: Colors.grey, width: 1.5),
                    ),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  onPressed: () => Navigator.of(context).pop(),
                  child: const Text('Cancel', style: TextStyle(fontWeight: FontWeight.bold)),
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red[100],
                    foregroundColor: Colors.red[900],
                    elevation: 4,
                    shadowColor: Colors.red[200],
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                      side: const BorderSide(color: Colors.red, width: 1.5),
                    ),
                    padding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                  onPressed: _isDeleting
                      ? null
                      : () async {
                          Navigator.of(context).pop();
                          await _deleteAccount();
                        },
                  child: _isDeleting
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Text('Yes, Delete', style: TextStyle(fontWeight: FontWeight.bold)),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _onNavTap(int index) {
    if (_selectedIndex == index) return; // Prevent unnecessary navigation
    Widget targetPage;
    int? targetIndex;
    if (index == 0) {
      targetPage = HomePage(onOpenSettings: widget.onOpenSettings);
    } else if (index == 1) {
      targetPage = ModeSelectionPage(onOpenSettings: widget.onOpenSettings);
      targetIndex = 1;
    } else if (index == 2) {
      targetPage = StatsPage(onOpenSettings: widget.onOpenSettings);
    } else if (index == 3) {
      // Already on Account
      return;
    } else {
      return;
    }
    Navigator.of(context).pushAndRemoveUntil(
      PageRouteBuilder(
        pageBuilder: (context, animation, secondaryAnimation) => targetIndex != null
            ? ModeSelectionPage(onOpenSettings: widget.onOpenSettings)
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

  void _navigateToFAQ() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => FAQScreen()),
    );
  }

  void _navigateToSettings() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (context) => SettingsScreen(onToggleTheme: widget.onOpenSettings)),
    );
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
    return Scaffold(
      backgroundColor: isDark ? const Color(0xFF1B2B1A) : Colors.white,
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
      body: Stack(
          children: [
          Container(
            decoration: isDark
                ? const BoxDecoration(color: Color(0xFF1B2B1A))
                : const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Color(0xFF9CB36B), Color(0xFFF5F5F5)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                  ),
          ),
          if (!isDark) ...[
            Positioned(
              top: -60,
              left: -60,
              child: Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  color: Colors.lightGreen[100]?.withOpacity(0.4),
                  shape: BoxShape.circle,
                ),
              ),
            ),
            Positioned(
              bottom: -40,
              right: -40,
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  color: Colors.brown[100]?.withOpacity(0.2),
                  shape: BoxShape.circle,
                ),
              ),
            ),
          ],
          SafeArea(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  // Profile Card
                  Padding(
                    padding: const EdgeInsets.fromLTRB(24, 32, 24, 16),
                    child: Card(
                      elevation: 10,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(28),
                      ),
                      color: cardColor,
                      child: Padding(
                        padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 24),
                child: Row(
                  children: [
                            Container(
                              width: 56,
                              height: 56,
                              decoration: BoxDecoration(
                                color: Colors.brown[700],
                                borderRadius: BorderRadius.circular(16),
                                image: _avatarUrl != null && _avatarUrl!.isNotEmpty
                                    ? DecorationImage(
                                        image: NetworkImage(_avatarUrl!),
                                        fit: BoxFit.cover,
                                      )
                                    : null,
                              ),
                              child: (_avatarUrl == null || _avatarUrl!.isEmpty)
                                  ? Center(
                      child: Text(
                        _isLoading
                            ? '?'
                            : _userName.isNotEmpty
                            ? _userName[0].toUpperCase()
                            : 'U',
                        style: const TextStyle(
                                          fontSize: 28,
                          fontWeight: FontWeight.bold,
                          color: Colors.white,
                        ),
                      ),
                                    )
                                  : null,
                    ),
                            const SizedBox(width: 18),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _isLoading ? 'Loading...' : _userName,
                                    style: TextStyle(
                                      fontSize: 22,
                              fontWeight: FontWeight.bold,
                                      color: textColor,
                            ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                          ),
                                  const SizedBox(height: 4),
                          Text(
                            _isLoading ? '' : _userEmail,
                            style: TextStyle(
                                      fontSize: 15,
                                      color: isDark ? Colors.white70 : Colors.grey[700],
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  if (_userDob != null)
                                    Padding(
                                      padding: const EdgeInsets.only(top: 6.0),
                                      child: Row(
                                        children: [
                                          Icon(Icons.cake_outlined, size: 18, color: isDark ? Colors.white70 : Colors.brown),
                                          const SizedBox(width: 6),
                                          Text(
                                            '${_userDob!.day.toString().padLeft(2, '0')}-${_userDob!.month.toString().padLeft(2, '0')}-${_userDob!.year}',
                                            style: TextStyle(
                                              fontSize: 15,
                                              color: isDark ? Colors.white70 : Colors.brown[700],
                                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                                ],
                                ),
                    ),
                  ],
                ),
              ),
            ),
                  ),
                  // Options List
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24.0),
              child: Column(
                children: [
                        _optionCard(
                          icon: Icons.edit,
                          label: 'Edit Profile',
                          onTap: _navigateToEditProfile,
                          color: isDark ? const Color(0xFF223D1B) : Colors.lightGreen[100],
                          textColor: textColor,
                  ),
                        const SizedBox(height: 16),
                        _optionCard(
                          icon: Icons.settings,
                          label: 'Settings',
                          onTap: _navigateToSettings,
                          color: isDark ? const Color(0xFF223D1B) : Colors.blueGrey[50],
                          textColor: textColor,
                  ),
                        const SizedBox(height: 16),
                        _optionCard(
                          icon: Icons.help_outline,
                          label: 'FAQ',
                          onTap: _navigateToFAQ,
                          color: isDark ? const Color(0xFF223D1B) : Colors.amber[50],
                          textColor: textColor,
                        ),
                        const SizedBox(height: 16),
                        _optionCard(
                          icon: Icons.logout,
                          label: 'Log Out',
                    onTap: _signOut,
                          color: isDark ? const Color(0xFF223D1B) : Colors.orange[50],
                          textColor: textColor,
                        ),
                        const SizedBox(height: 16),
                        _optionCard(
                          icon: Icons.delete,
                          label: 'Delete Account',
                          onTap: _showDeleteAccountDialog,
                          color: isDark ? const Color(0xFF223D1B) : Colors.red[50],
                          textColor: textColor,
                  ),
                ],
                    ),
                  ),
                  const SizedBox(height: 32),
                ],
              ),
              ),
            ),
          ],
      ),
    );
  }

  Widget _optionCard({required IconData icon, required String label, required VoidCallback onTap, Color? color, Color? textColor}) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
      ),
      color: color ?? Colors.white,
      child: ListTile(
        leading: Icon(icon, size: 28, color: textColor ?? Colors.brown[700]),
        title: Text(
          label,
          style: TextStyle(fontSize: 17, fontWeight: FontWeight.w600, color: textColor),
        ),
        onTap: onTap,
        trailing: Icon(Icons.arrow_forward_ios, size: 18, color: textColor?.withOpacity(0.7) ?? Colors.grey),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
        ),
      ),
    );
  }
}
