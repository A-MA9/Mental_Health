import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'package:permission_handler/permission_handler.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:intl/intl.dart';
import 'package:http/http.dart' as http;
import 'resultpage.dart';

class MentalHealthAssessment extends StatefulWidget {
  final CameraDescription camera;

  const MentalHealthAssessment({Key? key, required this.camera})
    : super(key: key);

  @override
  _MentalHealthAssessmentState createState() => _MentalHealthAssessmentState();
}

class _MentalHealthAssessmentState extends State<MentalHealthAssessment> {
  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;
  late FlutterSoundRecorder _audioRecorder;

  Timer? _photoTimer;
  bool _isRecording = false;
  String _userEmail = '';
  String _userId = '';
  String _currentAudioPath = '';
  String _currentSessionFolder = '';
  bool _isProcessing = false;

  int _currentQuestionIndex = 0;
  final List<String> _questions = [
    "How are you feeling today?",
    "Can you describe your current mood?",
    "Have you experienced feelings of anxiety recently?",
  ];

  String get currentQuestion => _questions[_currentQuestionIndex];

  @override
  void initState() {
    super.initState();

    // Request permissions
    _requestPermissions();

    // Initialize camera
    _cameraController = CameraController(
      widget.camera,
      ResolutionPreset.medium,
    );

    _initializeControllerFuture = _cameraController.initialize();

    // Initialize audio recorder
    _audioRecorder = FlutterSoundRecorder();
    _initAudioRecorder();

    // Get user info from Supabase
    _loadUserInfo();
  }

  void _loadUserInfo() {
    final user = Supabase.instance.client.auth.currentUser;
    if (user != null) {
      setState(() {
        _userEmail = user.email ?? "no-email@example.com";
        _userId = user.id;
      });
    } else {
      setState(() {
        _userEmail = "guest@example.com";
        _userId = "guest-user";
      });
    }
  }

  Future<void> _requestPermissions() async {
    await [
      Permission.camera,
      Permission.microphone,
      Permission.storage,
    ].request();
  }

  Future<void> _initAudioRecorder() async {
    await _audioRecorder.openRecorder();
  }

  void _startPhotoCapture() {
    _photoTimer = Timer.periodic(Duration(seconds: 2), (timer) {
      if (_cameraController.value.isInitialized && _isRecording) {
        _capturePhoto();
      }
    });
  }

  void _stopPhotoCapture() {
    _photoTimer?.cancel();
    _photoTimer = null;
  }

  Future<void> _capturePhoto() async {
    try {
      // Capture the image
      final XFile photo = await _cameraController.takePicture();

      // Upload the photo
      await _uploadPhotoToSupabase(photo.path);
    } catch (e) {
      print('Error taking photo: $e');
    }
  }

  Future<void> _startRecording() async {
    try {
      final directory = await getTemporaryDirectory();
      final dateFormatted = DateFormat('yyyy-MM-dd').format(DateTime.now());
      
      // Get the current question and sanitize it
      final currentQuestion = _questions[_currentQuestionIndex];
      final sanitizedQuestion = currentQuestion
          .replaceAll(' ', '_')
          .replaceAll('?', '');
      
      // Format: date_sanitizedQuestion
      _currentSessionFolder = '${dateFormatted}_${sanitizedQuestion}';
      final fileName = '${_currentSessionFolder}.aac';
      _currentAudioPath = '${directory.path}/$fileName';

      await _audioRecorder.startRecorder(
        toFile: _currentAudioPath,
        codec: Codec.aacADTS,
      );

      setState(() {
        _isRecording = true;
      });

      // Start taking photos
      _startPhotoCapture();
    } catch (e) {
      print('Error starting recording: $e');
    }
  }

  Future<void> _stopRecording() async {
    if (_isProcessing) return; // Prevent multiple calls

    try {
      // Stop photo capture
      _stopPhotoCapture();

      // Stop audio recording
      await _audioRecorder.stopRecorder();

      setState(() {
        _isRecording = false;
      });

      // Upload audio file
      await _uploadAudioToSupabase(_currentAudioPath);

      // Move to next question if available
      if (_currentQuestionIndex < _questions.length - 1) {
        setState(() {
          _currentQuestionIndex++;
        });
      } else {
        // All questions completed, navigate to results page
        setState(() {
          _isProcessing = true;
        });

        // Show processing indicator
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Recordings completed. Proceed to results page.',
            ),
            duration: Duration(seconds: 2),
          ),
        );

        // Navigate to results page - it will handle backend API calls
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(builder: (context) => ResultsPage()),
        );

        setState(() {
          _isProcessing = false;
        });
      }
    } catch (e) {
      print('Error stopping recording: $e');
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<void> _uploadPhotoToSupabase(String filePath) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) {
        print('File does not exist: $filePath');
        return;
      }

      final fileName = path.basename(filePath);

      // Format: userEmail/date_sanitizedQuestion/filename
      final storagePath = '${_userEmail}/${_currentSessionFolder}/${fileName}';

      await Supabase.instance.client.storage
          .from('interview-images')
          .upload(storagePath, file);
    } catch (e) {
      print('Error uploading photo: $e');
    }
  }

  Future<void> _uploadAudioToSupabase(String filePath) async {
    try {
      final file = File(filePath);
      if (!await file.exists()) {
        print('Audio file does not exist: $filePath');
        return;
      }

      final fileName = path.basename(filePath);

      // Format: userEmail/date_sanitizedQuestion.aac
      final storagePath = '${_userEmail}/${_currentSessionFolder}.aac';

      await Supabase.instance.client.storage
          .from('interview-audio')
          .upload(storagePath, file);
    } catch (e) {
      print('Error uploading audio: $e');
    }
  }

  @override
  void dispose() {
    _stopPhotoCapture();
    _cameraController.dispose();
    _audioRecorder.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Mental Health Assessment')),
      body: Column(
        children: [
          // Question section
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              currentQuestion,
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
          ),

          // Camera preview
          Expanded(
            child: FutureBuilder<void>(
              future: _initializeControllerFuture,
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.done) {
                  return CameraPreview(_cameraController);
                } else {
                  return Center(child: CircularProgressIndicator());
                }
              },
            ),
          ),

          // Record button
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: ElevatedButton(
              onPressed:
                  _isProcessing
                      ? null
                      : (_isRecording ? _stopRecording : _startRecording),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecording ? Colors.red : Colors.green,
                padding: EdgeInsets.symmetric(horizontal: 50, vertical: 15),
              ),
              child:
                  _isProcessing
                      ? Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          SizedBox(
                            width: 20,
                            height: 20,
                            child: CircularProgressIndicator(
                              color: Colors.white,
                            ),
                          ),
                          SizedBox(width: 10),
                          Text('Processing...', style: TextStyle(fontSize: 18)),
                        ],
                      )
                      : Text(
                        _isRecording ? 'Stop Recording' : 'Start Recording',
                        style: TextStyle(fontSize: 18),
                      ),
            ),
          ),
        ],
      ),
    );
  }
}

// Add AudioUploaderWidget to match what's used in model_selection.dart
class AudioUploaderWidget extends StatelessWidget {
  const AudioUploaderWidget({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<List<CameraDescription>>(
      future: availableCameras(),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          if (snapshot.hasData && snapshot.data!.isNotEmpty) {
            // Get front camera
            final frontCamera = snapshot.data!.firstWhere(
              (camera) => camera.lensDirection == CameraLensDirection.front,
              orElse: () => snapshot.data!.first,
            );

            return MentalHealthAssessment(camera: frontCamera);
          } else {
            return Scaffold(
              appBar: AppBar(title: Text('Error')),
              body: Center(child: Text('No camera available')),
            );
          }
        } else {
          return Scaffold(body: Center(child: CircularProgressIndicator()));
        }
      },
    );
  }
}
