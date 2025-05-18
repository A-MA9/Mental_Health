import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

class ResultsPage extends StatefulWidget {
  final Map<String, dynamic>? resultData;
  final String? userEmail;
  final String? userId;

  const ResultsPage({
    Key? key, 
    this.resultData,
    this.userEmail, 
    this.userId,
  }) : super(key: key);

  @override
  _ResultsPageState createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  bool _isLoading = false;
  String? _error;
  Map<String, dynamic>? _resultData;
  String? _userEmail;
  String? _userId;
  
  // Define multiple possible server addresses to try
  final List<String> apiUrls = [
    "http://172.16.63.49:8000/process_all_questions/",  // Your computer's IP address - best for physical devices
    "http://10.0.2.2:8000/process_all_questions/"      // Special Android emulator address that maps to host's localhost
    // "http://127.0.0.1:8000/process_all_questions/"      // Direct localhost - works on some simulators
  ];

  @override
  void initState() {
    super.initState();
    _resultData = widget.resultData;
    
    // Initialize user information
    _initUserInfo();
  }

  // Get user information directly from Supabase
  Future<void> _initUserInfo() async {
    // Use provided user info if available
    if (widget.userEmail != null && widget.userId != null) {
      _userEmail = widget.userEmail;
      _userId = widget.userId;
      
      // If we also have result data, we're done
      if (_resultData == null) {
        // Fetch results once we have user info
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _fetchResults();
        });
      }
      return;
    }
    
    // Otherwise get user info from Supabase
    try {
      final user = Supabase.instance.client.auth.currentUser;
      if (user != null) {
        setState(() {
          _userEmail = user.email ?? "no-email@example.com";
          _userId = user.id;
        });
        
        // Now fetch results with the user information
        if (_resultData == null) {
          _fetchResults();
        }
      } else {
        setState(() {
          _error = "Not logged in. Please log in to view your results.";
        });
      }
    } catch (e) {
      setState(() {
        _error = "Error retrieving user information: $e";
      });
    }
  }

  // Function to fetch results from the backend
  Future<void> _fetchResults() async {
    if (_userEmail == null || _userId == null) {
      setState(() {
        _error = "Missing user information. Cannot fetch results.";
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _error = null;
    });

    // Try each server address until one works
    for (String apiUrl in apiUrls) {
      try {
        print('Attempting to connect to: $apiUrl');
        
        // Set a timeout for the request
        final response = await http.post(
          Uri.parse(apiUrl),
          headers: {'Content-Type': 'application/json'},
          body: json.encode({
            'emailid': _userEmail,
            'uid': _userId,
          }),
        ).timeout(
          Duration(seconds: 90),  // Set a timeout of 15 seconds
          onTimeout: () {
            print('Connection to $apiUrl timed out');
            return http.Response('Timeout', 408);
          },
        );

        if (response.statusCode == 200) {
          print('Successfully connected to: $apiUrl');
          try {
            final data = json.decode(response.body);
            setState(() {
              _resultData = data;
              _isLoading = false;
            });
            return; // Success, exit function
          } catch (e) {
            print('Error parsing response: $e');
            continue; // Try next server if parsing fails
          }
        } else {
          print('API error at $apiUrl: ${response.statusCode} - ${response.body}');
          continue; // Try next server
        }
      } catch (e) {
        print('Error connecting to $apiUrl: $e');
        // Continue to next URL
      }
    }

    // If we reach here, all servers failed
    setState(() {
      _isLoading = false;
      _error = "Failed to connect to the server. Please check:\n"
             "1. Server is running on your computer\n"
             "2. Devices are on the same network\n"
             "3. No firewall is blocking the connection\n"
             "4. The server address is correct (${apiUrls[0]})";
    });
  }

  @override
  Widget build(BuildContext context) {
    // Show loading screen
    if (_isLoading) {
      return Scaffold(
        appBar: AppBar(title: const Text("Processing Results")),
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 20),
              Text("Analyzing your emotions...", style: TextStyle(fontSize: 16)),
              SizedBox(height: 10),
              Text("This may take a minute", style: TextStyle(fontSize: 14)),
            ],
          ),
        ),
      );
    }

    // Show error screen
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: const Text("Error")),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.error_outline, size: 60, color: Colors.red),
                const SizedBox(height: 20),
                Text(
                  _error!,
                  style: const TextStyle(fontSize: 16),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 30),
                ElevatedButton(
                  onPressed: _fetchResults,
                  child: const Text("Try Again"),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Show "get results" screen if no data but we have user info
    if (_resultData == null) {
      // Check if we have user info
      bool hasUserInfo = _userEmail != null && _userId != null;
      
      return Scaffold(
        appBar: AppBar(title: const Text("Results")),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.psychology, size: 80, color: Colors.blue),
                const SizedBox(height: 20),
                const Text(
                  "Your responses have been recorded",
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 10),
                const Text(
                  "Press the button below to analyze your emotional state",
                  style: TextStyle(fontSize: 16),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 30),
                ElevatedButton.icon(
                  icon: const Icon(Icons.analytics),
                  label: const Text("Get Results"),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40, vertical: 15),
                  ),
                  onPressed: hasUserInfo ? _fetchResults : null,
                ),
                if (!hasUserInfo)
                  const Padding(
                    padding: EdgeInsets.only(top: 16),
                    child: Text(
                      "User information not available. Please log in first.",
                      style: TextStyle(color: Colors.red),
                    ),
                  ),
              ],
            ),
          ),
        ),
      );
    }

    // Check for success status if we have data
    final bool success = _resultData!['success'] ?? false;
    
    if (!success) {
      // Show error message
      return Scaffold(
        appBar: AppBar(title: const Text("Analysis Error")),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(Icons.error_outline, size: 60, color: Colors.red),
                const SizedBox(height: 20),
                Text(
                  "Error analyzing emotions: ${_resultData!['error'] ?? 'Unknown error'}",
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 30),
                ElevatedButton(
                  onPressed: _fetchResults,
                  child: const Text("Try Again"),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // Get overall result
    final Map<String, dynamic> overall = _resultData!['overall'] ?? {};
    final String overallEmotion = overall['emotion'] ?? 'unknown';
    final double overallConfidence = overall['confidence'] ?? 0.0;
    
    // Get questions results
    final List<dynamic> questions = _resultData!['questions'] ?? [];

    return Scaffold(
      appBar: AppBar(
        title: const Text("Your Emotional Analysis"),
        backgroundColor: _getEmotionColor(overallEmotion),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Overall emotion card
              Card(
                elevation: 4,
                margin: const EdgeInsets.only(bottom: 24),
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        "Overall Analysis",
                        style: TextStyle(
                          fontSize: 22, 
                          fontWeight: FontWeight.bold
                        ),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Icon(
                            _getEmotionIcon(overallEmotion),
                            size: 48,
                            color: _getEmotionColor(overallEmotion),
                          ),
                          const SizedBox(width: 16),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                _capitalizeEmotion(overallEmotion),
                                style: const TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              Text(
                                "Confidence: ${(overallConfidence * 100).toStringAsFixed(1)}%",
                                style: TextStyle(
                                  fontSize: 16,
                                  color: Colors.grey[600],
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        _getEmotionDescription(overallEmotion),
                        style: const TextStyle(fontSize: 16),
                      ),
                    ],
                  ),
                ),
              ),
              
              // Per-question analysis
              const Text(
                "Question-by-Question Analysis",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              
              // Question cards
              ...questions.map((question) => _buildQuestionCard(question)).toList(),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildQuestionCard(Map<String, dynamic> question) {
    final String questionText = question['question'] ?? 'Unknown question';
    final Map<String, dynamic> aggregate = question['aggregate'] ?? {};
    final String emotion = aggregate['emotion'] ?? 'unknown';
    final double confidence = aggregate['confidence'] ?? 0.0;
    final String transcript = question['transcript'] ?? '';
    
    // Get individual modality results
    final Map<String, dynamic> faceResult = question['face'] ?? {};
    final Map<String, dynamic> audioResult = question['audio'] ?? {};
    final Map<String, dynamic> textResult = question['text'] ?? {};
    
    final String faceEmotion = faceResult['emotion'] ?? 'unknown';
    final String audioEmotion = audioResult['emotion'] ?? 'unknown';
    final String textEmotion = textResult['emotion'] ?? 'unknown';
    
    return Card(
      elevation: 3,
      margin: const EdgeInsets.only(bottom: 16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              questionText,
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const Divider(),
            Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildEmotionItem(
                        "🎭 Face", 
                        faceEmotion, 
                        _getEmotionColor(faceEmotion)
                      ),
                      _buildEmotionItem(
                        "🔊 Voice", 
                        audioEmotion, 
                        _getEmotionColor(audioEmotion)
                      ),
                      _buildEmotionItem(
                        "🧠 Words", 
                        textEmotion, 
                        _getEmotionColor(textEmotion)
                      ),
                    ],
                  ),
                ),
                Column(
                  children: [
                    Icon(
                      _getEmotionIcon(emotion),
                      size: 40,
                      color: _getEmotionColor(emotion),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      _capitalizeEmotion(emotion),
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: _getEmotionColor(emotion),
                      ),
                    ),
                    Text(
                      "${(confidence * 100).toStringAsFixed(0)}%",
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ],
            ),
            if (transcript.isNotEmpty) ...[
              const Divider(),
              const Text(
                "Your Response:",
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                transcript,
                style: TextStyle(
                  fontStyle: FontStyle.italic,
                  color: Colors.grey[700],
                  fontSize: 14,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
  
  Widget _buildEmotionItem(String label, String emotion, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(
            label + ": ",
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          Text(
            _capitalizeEmotion(emotion),
            style: TextStyle(color: color),
          ),
        ],
      ),
    );
  }
  
  String _capitalizeEmotion(String emotion) {
    if (emotion.isEmpty) return "Unknown";
    return emotion[0].toUpperCase() + emotion.substring(1);
  }
  
  IconData _getEmotionIcon(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
      case 'joy':
        return Icons.sentiment_very_satisfied;
      case 'sad':
      case 'sadness':
        return Icons.sentiment_dissatisfied;
      case 'angry':
      case 'anger':
        return Icons.sentiment_very_dissatisfied;
      case 'fear':
        return Icons.psychology;
      case 'surprise':
        return Icons.emoji_emotions;
      case 'disgust':
        return Icons.sick;
      case 'neutral':
        return Icons.sentiment_neutral;
      case 'love':
        return Icons.favorite;
      default:
        return Icons.face;
    }
  }
  
  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
      case 'joy':
        return Colors.amber;
      case 'sad':
      case 'sadness':
        return Colors.blue;
      case 'angry':
      case 'anger':
        return Colors.red;
      case 'fear':
        return Colors.purple;
      case 'surprise':
        return Colors.orange;
      case 'disgust':
        return Colors.green;
      case 'neutral':
        return Colors.grey;
      case 'love':
        return Colors.pink;
      default:
        return Colors.teal;
    }
  }
  
  String _getEmotionDescription(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
      case 'joy':
        return "You appear to be in a positive, upbeat mood. This is great for your mental wellbeing!";
      case 'sad':
      case 'sadness':
        return "You seem to be feeling down. Remember it's okay to feel this way sometimes. Consider talking to someone you trust.";
      case 'angry':
      case 'anger':
        return "You're showing signs of frustration or anger. Try some deep breathing or a calming activity.";
      case 'fear':
        return "You appear to be experiencing anxiety or fear. Consider grounding techniques to help center yourself.";
      case 'surprise':
        return "You're showing expressions of surprise or astonishment.";
      case 'disgust':
        return "You're displaying signs of aversion or displeasure.";
      case 'neutral':
        return "Your emotional state appears balanced and steady.";
      case 'love':
        return "You're expressing warmth and affection.";
      default:
        return "Unable to determine a specific emotional pattern.";
    }
  }
}
