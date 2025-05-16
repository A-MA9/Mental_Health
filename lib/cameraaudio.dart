// import 'dart:async';
// import 'dart:io';
// import 'package:camera/camera.dart';
// import 'package:flutter/material.dart';
// import 'package:path/path.dart' as p;
// import 'package:path_provider/path_provider.dart';
// import 'package:permission_handler/permission_handler.dart';
// import 'package:supabase_flutter/supabase_flutter.dart';
// import 'package:flutter_sound/flutter_sound.dart';
// import 'package:audio_session/audio_session.dart';

// class VideoQuestionnairePage extends StatefulWidget {
//   final bool isDarkMode;
//   final VoidCallback toggleTheme;

//   const VideoQuestionnairePage({
//     super.key,
//     required this.isDarkMode,
//     required this.toggleTheme,
//   });

//   @override
//   State<VideoQuestionnairePage> createState() => _VideoQuestionnairePageState();
// }

// class _VideoQuestionnairePageState extends State<VideoQuestionnairePage> {
//   List<String> questions = [
//     "How are you feeling today?",
//     "Can you describe your current mood?",
//     "What made you feel this way?",
//   ];

//   int currentIndex = 0;
//   CameraController? _controller;
//   bool _isCameraInitialized = false;
//   bool _isRecording = false;
//   Timer? _timer;
//   List<File> capturedImages = [];

//   // Audio recording properties
//   FlutterSoundRecorder? _audioRecorder;
//   bool _isAudioRecorderInitialized = false;
//   String? _audioPath;

//   @override
//   void initState() {
//     super.initState();
//     _initCamera();
//     _initAudioRecorder();
//   }

//   Future<void> _initCamera() async {
//     final status = await Permission.camera.request();
//     if (status.isGranted) {
//       final cameras = await availableCameras();
//       final frontCamera = cameras.firstWhere(
//         (c) => c.lensDirection == CameraLensDirection.front,
//       );
//       _controller = CameraController(frontCamera, ResolutionPreset.medium);
//       await _controller!.initialize();
//       if (mounted) {
//         setState(() => _isCameraInitialized = true);
//       }
//     } else {
//       ScaffoldMessenger.of(context).showSnackBar(
//         const SnackBar(content: Text("Camera permission is required.")),
//       );
//     }
//   }

//   Future<void> _initAudioRecorder() async {
//     final status = await Permission.microphone.request();
//     if (status.isGranted) {
//       _audioRecorder = FlutterSoundRecorder();
//       await _audioRecorder!.openRecorder();

//       // Set up audio session
//       final session = await AudioSession.instance;
//       await session.configure(AudioSessionConfiguration(
//         avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
//         avAudioSessionCategoryOptions: AVAudioSessionCategoryOptions.allowBluetooth,
//         avAudioSessionMode: AVAudioSessionMode.spokenAudio,
//         avAudioSessionRouteSharingPolicy: AVAudioSessionRouteSharingPolicy.defaultPolicy,
//         avAudioSessionSetActiveOptions: AVAudioSessionSetActiveOptions.none,
//         androidAudioAttributes: const AndroidAudioAttributes(
//           contentType: AndroidAudioContentType.speech,
//           flags: AndroidAudioFlags.none,
//           usage: AndroidAudioUsage.voiceCommunication,
//         ),
//         androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
//         androidWillPauseWhenDucked: true,
//       ));

//       setState(() => _isAudioRecorderInitialized = true);
//     } else {
//       ScaffoldMessenger.of(context).showSnackBar(
//         const SnackBar(content: Text("Microphone permission is required.")),
//       );
//     }
//   }

//   Future<void> _startRecording() async {
//     if (_controller == null || !_controller!.value.isInitialized) return;
//     if (_audioRecorder == null || !_isAudioRecorderInitialized) return;

//     setState(() => _isRecording = true);

//     // Start audio recording
//     final directory = await getTemporaryDirectory();
//     final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
//     _audioPath = '${directory.path}/audio_$timestamp.aac';
//     await _audioRecorder!.startRecorder(
//       toFile: _audioPath,
//       codec: Codec.aacADTS,
//     );

//     // Start image capture timer
//     _timer = Timer.periodic(const Duration(seconds: 2), (timer) async {
//       try {
//         final path = await _takePicture();
//         if (path != null) {
//           capturedImages.add(File(path));
//         }
//       } catch (e) {
//         debugPrint("Capture error: $e");
//       }
//     });
//   }

//   Future<void> _stopRecording() async {
//     // Stop image capture timer
//     _timer?.cancel();

//     // Stop audio recording
//     String? audioFilePath;
//     if (_audioRecorder != null && _audioRecorder!.isRecording) {
//       audioFilePath = await _audioRecorder!.stopRecorder();
//     }

//     setState(() => _isRecording = false);

//     final question = questions[currentIndex].replaceAll(" ", "_");
//     final timestamp = DateTime.now().toIso8601String().replaceAll(":", "-");
//     final folderName = "$question-$timestamp";

//     // Upload images to Supabase
//     await _uploadImagesToSupabase(folderName, capturedImages);

//     // Upload audio to Supabase if available
//     if (audioFilePath != null) {
//       await _uploadAudioToSupabase(folderName, File(audioFilePath));
//     }

//     capturedImages.clear();
//     setState(() => currentIndex++);
//   }

//   Future<String?> _takePicture() async {
//     if (!_controller!.value.isInitialized) return null;

//     final directory = await getTemporaryDirectory();
//     final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
//     final filePath = p.join(directory.path, 'image_$timestamp.jpg');

//     try {
//       await _controller!.takePicture().then((XFile file) async {
//         final savedImage = await File(file.path).copy(filePath);
//         return savedImage.path;
//       });
//       return filePath;
//     } catch (e) {
//       debugPrint("Take picture failed: $e");
//       return null;
//     }
//   }

//   Future<void> _uploadImagesToSupabase(
//     String folderName,
//     List<File> files,
//   ) async {
//     final supabase = Supabase.instance.client;

//     for (var file in files) {
//       final fileName = p.basename(file.path);
//       final storagePath = '$folderName/$fileName';

//       try {
//         await supabase.storage
//             .from('interview-images') // your bucket name
//             .upload(storagePath, file);
//       } catch (e) {
//         debugPrint('Upload failed for $fileName: $e');
//         ScaffoldMessenger.of(
//           context,
//         ).showSnackBar(SnackBar(content: Text("Upload failed for $fileName")));
//       }
//     }

//     ScaffoldMessenger.of(context).showSnackBar(
//       const SnackBar(content: Text("Images uploaded successfully.")),
//     );
//   }

//   Future<void> _uploadAudioToSupabase(
//     String folderName,
//     File audioFile,
//   ) async {
//     final supabase = Supabase.instance.client;
//     final fileName = p.basename(audioFile.path);
//     final storagePath = '$folderName/$fileName';

//     try {
//       await supabase.storage
//           .from('interview-audio') // your new audio bucket name
//           .upload(storagePath, audioFile);

//       ScaffoldMessenger.of(context).showSnackBar(
//         const SnackBar(content: Text("Audio uploaded successfully.")),
//       );
//     } catch (e) {
//       debugPrint('Audio upload failed: $e');
//       ScaffoldMessenger.of(context).showSnackBar(
//         SnackBar(content: Text("Audio upload failed: ${e.toString()}")),
//       );
//     }
//   }

//   @override
//   void dispose() {
//     _timer?.cancel();
//     _controller?.dispose();
//     _audioRecorder?.closeRecorder();
//     super.dispose();
//   }

//   @override
//   Widget build(BuildContext context) {
//     final isLastQuestion = currentIndex >= questions.length;
//     final themeColor = widget.isDarkMode ? Colors.black : Colors.white;
//     final textColor = widget.isDarkMode ? Colors.white : Colors.black;

//     return Scaffold(
//       backgroundColor: themeColor,
//       appBar: AppBar(
//         title: const Text("Mental Health Interview"),
//         actions: [
//           IconButton(
//             icon: AnimatedSwitcher(
//               duration: const Duration(milliseconds: 300),
//               transitionBuilder:
//                   (child, animation) =>
//                       RotationTransition(turns: animation, child: child),
//               child:
//                   widget.isDarkMode
//                       ? const Icon(Icons.dark_mode, key: ValueKey('dark'))
//                       : const Icon(Icons.wb_sunny, key: ValueKey('light')),
//             ),
//             onPressed: widget.toggleTheme,
//           ),
//         ],
//       ),
//       body:
//           _isCameraInitialized
//               ? Column(
//                 children: [
//                   if (!isLastQuestion)
//                     Expanded(
//                       child: Column(
//                         children: [
//                           Container(
//                             padding: const EdgeInsets.all(16),
//                             margin: const EdgeInsets.all(12),
//                             decoration: BoxDecoration(
//                               color:
//                                   widget.isDarkMode
//                                       ? Colors.grey[800]
//                                       : Colors.green[100],
//                               borderRadius: BorderRadius.circular(20),
//                             ),
//                             child: Text(
//                               questions[currentIndex],
//                               style: TextStyle(
//                                 fontSize: 18,
//                                 fontWeight: FontWeight.w500,
//                                 color: textColor,
//                               ),
//                             ),
//                           ),
//                           Expanded(child: CameraPreview(_controller!)),
//                         ],
//                       ),
//                     )
//                   else
//                     Expanded(
//                       child: Center(
//                         child: ElevatedButton.icon(
//                           icon: const Icon(Icons.analytics),
//                           label: const Text("View Result"),
//                           style: ElevatedButton.styleFrom(
//                             minimumSize: const Size(200, 50),
//                             backgroundColor: Colors.teal,
//                           ),
//                           onPressed: () {
//                             // Navigator.push(context, MaterialPageRoute(
//                             //   builder: (_) => const ResultPage(),
//                             // ));
//                           },
//                         ),
//                       ),
//                     ),
//                   const SizedBox(height: 16),
//                   if (!isLastQuestion)
//                     Padding(
//                       padding: const EdgeInsets.all(16.0),
//                       child: Row(
//                         children: [
//                           Expanded(
//                             child: ElevatedButton.icon(
//                               icon: Icon(
//                                 _isRecording ? Icons.stop : Icons.fiber_manual_record,
//                               ),
//                               label: Text(
//                                 _isRecording ? "Stop Recording" : "Start Recording",
//                               ),
//                               style: ElevatedButton.styleFrom(
//                                 backgroundColor:
//                                     _isRecording ? Colors.red : Colors.green,
//                                 minimumSize: const Size(double.infinity, 50),
//                               ),
//                               onPressed:
//                                   _isRecording ? _stopRecording : _startRecording,
//                             ),
//                           ),
//                         ],
//                       ),
//                     ),
//                 ],
//               )
//               : const Center(child: CircularProgressIndicator()),
//     );
//   }
// }

import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:audio_session/audio_session.dart';
import 'package:intl/intl.dart'; // <-- Add this for date formatting

class VideoQuestionnairePage extends StatefulWidget {
  final VoidCallback toggleTheme;

  const VideoQuestionnairePage({
    super.key,
    required this.toggleTheme,
  });

  @override
  State createState() => _VideoQuestionnairePageState();
}

class _VideoQuestionnairePageState extends State<VideoQuestionnairePage> {
  List<String> questions = [
    "How are you feeling today?",
    "Can you describe your current mood?",
    "What made you feel this way?",
  ];
  int currentIndex = 0;
  CameraController? _controller;
  bool _isCameraInitialized = false;
  bool _isRecording = false;
  Timer? _timer;
  List<File> capturedImages = [];

  // Audio recording properties
  FlutterSoundRecorder? _audioRecorder;
  bool _isAudioRecorderInitialized = false;
  String? _audioPath;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initAudioRecorder();
  }

  Future _initCamera() async {
    final status = await Permission.camera.request();
    if (status.isGranted) {
      final cameras = await availableCameras();
      final frontCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
      );
      _controller = CameraController(frontCamera, ResolutionPreset.medium);
      await _controller!.initialize();
      if (mounted) {
        setState(() => _isCameraInitialized = true);
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Camera permission is required.")),
      );
    }
  }

  Future _initAudioRecorder() async {
    final status = await Permission.microphone.request();
    if (status.isGranted) {
      _audioRecorder = FlutterSoundRecorder();
      await _audioRecorder!.openRecorder();
      // Set up audio session
      final session = await AudioSession.instance;
      await session.configure(
        AudioSessionConfiguration(
          avAudioSessionCategory: AVAudioSessionCategory.playAndRecord,
          avAudioSessionCategoryOptions:
              AVAudioSessionCategoryOptions.allowBluetooth,
          avAudioSessionMode: AVAudioSessionMode.spokenAudio,
          avAudioSessionRouteSharingPolicy:
              AVAudioSessionRouteSharingPolicy.defaultPolicy,
          avAudioSessionSetActiveOptions: AVAudioSessionSetActiveOptions.none,
          androidAudioAttributes: const AndroidAudioAttributes(
            contentType: AndroidAudioContentType.speech,
            flags: AndroidAudioFlags.none,
            usage: AndroidAudioUsage.voiceCommunication,
          ),
          androidAudioFocusGainType: AndroidAudioFocusGainType.gain,
          androidWillPauseWhenDucked: true,
        ),
      );
      setState(() => _isAudioRecorderInitialized = true);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Microphone permission is required.")),
      );
    }
  }

  Future _startRecording() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_audioRecorder == null || !_isAudioRecorderInitialized) return;
    setState(() => _isRecording = true);

    // Start audio recording
    final directory = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    _audioPath = '${directory.path}/audio_$timestamp.aac';
    await _audioRecorder!.startRecorder(
      toFile: _audioPath,
      codec: Codec.aacADTS,
    );

    // Start image capture timer
    _timer = Timer.periodic(const Duration(seconds: 2), (timer) async {
      try {
        final path = await _takePicture();
        if (path != null) {
          capturedImages.add(File(path));
        }
      } catch (e) {
        debugPrint("Capture error: $e");
      }
    });
  }

  Future _stopRecording() async {
    // Stop image capture timer
    _timer?.cancel();

    // Stop audio recording
    String? audioFilePath;
    if (_audioRecorder != null && _audioRecorder!.isRecording) {
      audioFilePath = await _audioRecorder!.stopRecorder();
    }
    setState(() => _isRecording = false);

    // --- CHANGES START HERE ---
    // Get user email
    final user = Supabase.instance.client.auth.currentUser;
    final userEmail = user?.email ?? 'unknown_user';

    // Format date and sanitize question
    final now = DateTime.now();
    final formattedDate = DateFormat('yyyy-MM-dd').format(now);
    final rawQuestion = questions[currentIndex];
    final sanitizedQuestion = rawQuestion
        .replaceAll(' ', '_')
        .replaceAll(RegExp(r'[^\w\-]'), '');

    // Use: userEmail/yyyy-MM-dd_sanitizedQuestion/filename
    final folderName = "$userEmail/${formattedDate}_$sanitizedQuestion";

    // Upload images to Supabase
    await _uploadImagesToSupabase(folderName, capturedImages);

    // Upload audio to Supabase if available
    if (audioFilePath != null) {
      await _uploadAudioToSupabase(folderName, File(audioFilePath));
    }

    capturedImages.clear();
    setState(() => currentIndex++);
    // --- CHANGES END HERE ---
  }

  Future<String?> _takePicture() async {
    if (!_controller!.value.isInitialized) return null;
    final directory = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch.toString();
    final filePath = p.join(directory.path, 'image_$timestamp.jpg');
    try {
      await _controller!.takePicture().then((XFile file) async {
        final savedImage = await File(file.path).copy(filePath);
        return savedImage.path;
      });
      return filePath;
    } catch (e) {
      debugPrint("Take picture failed: $e");
      return null;
    }
  }

  // --- CHANGES START HERE ---
  Future _uploadImagesToSupabase(String folderName, List<File> files) async {
    final supabase = Supabase.instance.client;
    for (var file in files) {
      final fileName = p.basename(file.path);
      final storagePath = '$folderName/$fileName';
      try {
        await supabase.storage
            .from('interview-images')
            .upload(storagePath, file);
      } catch (e) {
        debugPrint('Upload failed for $fileName: $e');
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Upload failed for $fileName")));
      }
    }
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text("Images uploaded successfully.")),
    );
  }

  Future _uploadAudioToSupabase(String folderName, File audioFile) async {
    final supabase = Supabase.instance.client;
    final fileName = p.basename(audioFile.path);
    final storagePath = '$folderName/$fileName';
    try {
      await supabase.storage
          .from('interview-audio')
          .upload(storagePath, audioFile);
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Audio uploaded successfully.")),
      );
    } catch (e) {
      debugPrint('Audio upload failed: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Audio upload failed: ${e.toString()}")),
      );
    }
  }
  // --- CHANGES END HERE ---

  @override
  void dispose() {
    _timer?.cancel();
    _controller?.dispose();
    _audioRecorder?.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final isLastQuestion = currentIndex >= questions.length;
    final themeColor = isDark ? Colors.black : Colors.white;
    final textColor = isDark ? Colors.white : Colors.black;

    return Scaffold(
      backgroundColor: themeColor,
      appBar: AppBar(
        title: const Text("Mental Health Interview"),
        actions: [
          IconButton(
            icon: AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              transitionBuilder:
                  (child, animation) =>
                      RotationTransition(turns: animation, child: child),
              child:
                  isDark
                      ? const Icon(Icons.dark_mode, key: ValueKey('dark'))
                      : const Icon(Icons.wb_sunny, key: ValueKey('light')),
            ),
            onPressed: widget.toggleTheme,
          ),
        ],
      ),
      body:
          _isCameraInitialized
              ? Column(
                children: [
                  if (!isLastQuestion)
                    Expanded(
                      child: Column(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(16),
                            margin: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color:
                                  isDark
                                      ? Colors.grey[800]
                                      : Colors.green[100],
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              questions[currentIndex],
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.w500,
                                color: textColor,
                              ),
                            ),
                          ),
                          Expanded(child: CameraPreview(_controller!)),
                        ],
                      ),
                    )
                  else
                    Expanded(
                      child: Center(
                        child: ElevatedButton.icon(
                          icon: const Icon(Icons.analytics),
                          label: const Text("View Result"),
                          style: ElevatedButton.styleFrom(
                            minimumSize: const Size(200, 50),
                            backgroundColor: Colors.teal,
                          ),
                          onPressed: () {
                            // Navigator.push(context, MaterialPageRoute(
                            //   builder: (_) => const ResultPage(),
                            // ));
                          },
                        ),
                      ),
                    ),
                  const SizedBox(height: 16),
                  if (!isLastQuestion)
                    Padding(
                      padding: const EdgeInsets.all(16.0),
                      child: Row(
                        children: [
                          Expanded(
                            child: ElevatedButton.icon(
                              icon: Icon(
                                _isRecording
                                    ? Icons.stop
                                    : Icons.fiber_manual_record,
                              ),
                              label: Text(
                                _isRecording
                                    ? "Stop Recording"
                                    : "Start Recording",
                              ),
                              style: ElevatedButton.styleFrom(
                                backgroundColor:
                                    _isRecording ? Colors.red : Colors.green,
                                minimumSize: const Size(double.infinity, 50),
                              ),
                              onPressed:
                                  _isRecording
                                      ? _stopRecording
                                      : _startRecording,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              )
              : const Center(child: CircularProgressIndicator()),
    );
  }
}
