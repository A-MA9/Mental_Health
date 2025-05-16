import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'permissions_handler.dart';
import 'dart:io' show Platform;
import 'package:flutter/foundation.dart';

class PermissionsScreen extends StatefulWidget {
  final VoidCallback onPermissionsGranted;
  const PermissionsScreen({Key? key, required this.onPermissionsGranted}) : super(key: key);

  @override
  State<PermissionsScreen> createState() => _PermissionsScreenState();
}

class _PermissionsScreenState extends State<PermissionsScreen> {
  bool _isCameraGranted = false;
  bool _isMicrophoneGranted = false;
  bool _isStorageGranted = false;
  bool _isNotificationGranted = false;
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _checkPermissions();
  }

  Future<void> _checkPermissions() async {
    setState(() => _isLoading = true);
    
    // Check camera permission
    _isCameraGranted = await Permission.camera.isGranted;
    
    // Check microphone permission
    _isMicrophoneGranted = await Permission.microphone.isGranted;
    
    // Check storage permissions
    if (Platform.isAndroid) {
      _isStorageGranted = await Permission.storage.isGranted || 
                         (await Permission.photos.isGranted && 
                          await Permission.videos.isGranted &&
                          await Permission.audio.isGranted);
    } else {
      _isStorageGranted = await Permission.storage.isGranted;
    }
    
    // Check notification permission
    _isNotificationGranted = await Permission.notification.isGranted;
    
    setState(() => _isLoading = false);
    
    // If all permissions are already granted, navigate to the next screen
    if (_isCameraGranted && _isMicrophoneGranted && _isStorageGranted && _isNotificationGranted) {
      widget.onPermissionsGranted();
    }
  }

  Future<void> _requestAllPermissions() async {
    setState(() => _isLoading = true);
    
    // Request each permission explicitly, one by one
    // This way we ensure each permission shows its own system dialog
    
    // 1. Request camera permission
    await Permission.camera.request();
    _isCameraGranted = await Permission.camera.isGranted;
    
    // 2. Request microphone permission
    await Permission.microphone.request();
    _isMicrophoneGranted = await Permission.microphone.isGranted;
    
    // 3. Request storage permissions - multiple for Android 13+
    await Permission.storage.request();
    if (Platform.isAndroid) {
      await Permission.photos.request();
      await Permission.videos.request();
      await Permission.audio.request();
      
      _isStorageGranted = await Permission.storage.isGranted || 
                          (await Permission.photos.isGranted && 
                           await Permission.videos.isGranted);
    } else {
      _isStorageGranted = await Permission.storage.isGranted;
    }
    
    // 4. Request notification permission separately
    // Small delay to ensure previous permission dialogs are closed
    await Future.delayed(const Duration(milliseconds: 500));
    await Permission.notification.request();
    _isNotificationGranted = await Permission.notification.isGranted;
    
    setState(() => _isLoading = false);
    
    // Check if critical permissions are granted
    if (_isCameraGranted && _isMicrophoneGranted && _isStorageGranted) {
      // Even if notification permission isn't granted, we can still proceed
      widget.onPermissionsGranted();
    }
  }

  // Individual permission request methods
  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    setState(() => _isCameraGranted = status.isGranted);
  }

  Future<void> _requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    setState(() => _isMicrophoneGranted = status.isGranted);
  }

  Future<void> _requestStoragePermission() async {
    // For storage, we need to request multiple permissions on newer Android
    await Permission.storage.request();
    
    if (Platform.isAndroid) {
      // On Android 13+, request specific media permissions
      await Permission.photos.request();
      await Permission.videos.request();
      await Permission.audio.request();
      await Permission.mediaLibrary.request();
      
      final isGranted = await Permission.storage.isGranted || 
                        (await Permission.photos.isGranted && 
                         await Permission.videos.isGranted &&
                         await Permission.audio.isGranted);
      setState(() => _isStorageGranted = isGranted);
    } else {
      final isGranted = await Permission.storage.isGranted;
      setState(() => _isStorageGranted = isGranted);
    }
  }

  Future<void> _requestNotificationPermission() async {
    final status = await Permission.notification.request();
    setState(() => _isNotificationGranted = status.isGranted);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF1B2B1A) : Colors.white;
    final cardColor = isDark ? const Color(0xFF223D1B) : Colors.white;
    final textColor = isDark ? Colors.white : const Color(0xFF3A2713);
    return Scaffold(
      backgroundColor: bgColor,
      body: _isLoading 
        ? const Center(child: CircularProgressIndicator(color: Colors.green))
        : SafeArea(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [
                    const Color(0xFFE8F5E9), // very light green
                    const Color(0xFFD0F5C3), // another light green
                  ],
                ),
              ),
              child: CustomPaint(
                painter: _TexturePainter(),
                child: Padding(
                  padding: const EdgeInsets.all(24.0),
                  child: SingleChildScrollView(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const SizedBox(height: 24),
                        // App Logo
                        Center(
                          child: Container(
                            padding: const EdgeInsets.all(15),
                            decoration: BoxDecoration(
                              color: isDark ? const Color(0xFF223D1B) : Colors.white,
                              shape: BoxShape.circle,
                              boxShadow: [BoxShadow(color: Colors.black12, blurRadius: 10)],
                            ),
                            child: Image.asset('assets/logo.png', width: 80, height: 80),
                          ),
                        ),
                        const SizedBox(height: 16),
                        // App Permissions Title
                        Text(
                          "App Permissions",
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: textColor,
                            letterSpacing: 0.5,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 12),
                        // Description
                        Text(
                          "मनन needs the following permissions to provide you with the best experience:",
                          style: TextStyle(
                            fontSize: 16,
                            color: isDark ? Colors.white70 : Colors.black87,
                          ),
                          textAlign: TextAlign.center,
                        ),
                        const SizedBox(height: 20),
                        // Permissions List
                        PermissionItem(
                          title: "Camera",
                          description: "To allow you to capture photos and videos",
                          icon: Icons.camera_alt_outlined,
                          isGranted: _isCameraGranted,
                          onRequest: _requestCameraPermission,
                          isDark: isDark,
                        ),
                        PermissionItem(
                          title: "Microphone",
                          description: "To record your voice for mood analysis",
                          icon: Icons.mic_outlined,
                          isGranted: _isMicrophoneGranted,
                          onRequest: _requestMicrophonePermission,
                          isDark: isDark,
                        ),
                        PermissionItem(
                          title: "Storage",
                          description: "To save your recordings and media",
                          icon: Icons.folder_outlined,
                          isGranted: _isStorageGranted,
                          onRequest: _requestStoragePermission,
                          isDark: isDark,
                        ),
                        PermissionItem(
                          title: "Notifications",
                          description: "To remind you about your wellness activities",
                          icon: Icons.notifications_outlined,
                          isGranted: _isNotificationGranted,
                          onRequest: _requestNotificationPermission,
                          isDark: isDark,
                        ),
                        const SizedBox(height: 32),
                        // Grant All Permissions Button
                        ElevatedButton(
                          onPressed: _requestAllPermissions,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF9CB36B),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(30),
                            ),
                          ),
                          child: const Text(
                            "Grant All Permissions",
                            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                          ),
                        ),
                        const SizedBox(height: 24),
                        // Continue Button with 3D white rounded boundary
                        Container(
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(30),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black12,
                                blurRadius: 10,
                                offset: Offset(0, 4),
                              ),
                            ],
                          ),
                          child: TextButton(
                            onPressed: () {
                              if (_isCameraGranted && _isMicrophoneGranted && _isStorageGranted) {
                                widget.onPermissionsGranted();
                              } else {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text("Camera, Microphone and Storage permissions are required"),
                                    duration: Duration(seconds: 3),
                                  ),
                                );
                              }
                            },
                            style: TextButton.styleFrom(
                              minimumSize: const Size.fromHeight(48),
                              padding: const EdgeInsets.symmetric(vertical: 16),
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(30),
                              ),
                              foregroundColor: textColor,
                            ),
                            child: Text(
                              "Continue",
                              style: TextStyle(
                                color: textColor,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 24),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
    );
  }
}

class PermissionItem extends StatelessWidget {
  final String title;
  final String description;
  final IconData icon;
  final bool isGranted;
  final VoidCallback onRequest;
  final bool isDark;

  const PermissionItem({
    Key? key,
    required this.title,
    required this.description,
    required this.icon,
    required this.isGranted,
    required this.onRequest,
    required this.isDark,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16.0),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: isGranted ? Colors.green.withOpacity(0.1) : (isDark ? Colors.white10 : Colors.grey.withOpacity(0.1)),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(
              icon,
              color: isGranted ? Colors.green : (isDark ? Colors.white : Colors.grey),
              size: 24,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: isDark ? Colors.white : Colors.black,
                  ),
                ),
                Text(
                  description,
                  style: TextStyle(
                    fontSize: 14,
                    color: isDark ? Colors.white70 : Colors.grey[600],
                  ),
                ),
              ],
            ),
          ),
          TextButton(
            onPressed: onRequest,
            style: TextButton.styleFrom(
              foregroundColor: isGranted ? Colors.green : const Color(0xFF9CB36B),
              padding: const EdgeInsets.symmetric(horizontal: 8),
            ),
            child: Text(isGranted ? "Granted" : "Request"),
          ),
        ],
      ),
    );
  }
}

class _TexturePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0x22A5D6A7) // semi-transparent green
      ..style = PaintingStyle.fill;
    // Draw subtle diagonal lines for 3D texture
    for (double y = 0; y < size.height; y += 32) {
      canvas.drawRect(
        Rect.fromLTWH(-size.width * 0.2, y, size.width * 1.4, 2),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
} 