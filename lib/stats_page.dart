import 'package:flutter/material.dart';
import 'package:curved_navigation_bar/curved_navigation_bar.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:intl/intl.dart';
import 'homepage.dart';
import 'model_selection.dart';
import 'account.dart';
import 'main.dart';

class StatsPage extends StatefulWidget {
  final VoidCallback onOpenSettings;
  const StatsPage({Key? key, required this.onOpenSettings}) : super(key: key);

  @override
  State<StatsPage> createState() => _StatsPageState();
}

class _StatsPageState extends State<StatsPage> {
  int _selectedIndex = 2; // Stats tab
  bool _isLoading = true;
  Map<String, List<Map<String, dynamic>>> _emotionDataByDate = {};
  String? _error;
  Set<String> _expandedCards = {}; // Track which cards are expanded

  @override
  void initState() {
    super.initState();
    _fetchEmotionData();
  }

  Future<void> _fetchEmotionData() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final user = supabase.auth.currentUser;
      if (user == null) {
        throw Exception('User not authenticated');
      }

      // Fetch emotion analysis data for the current user
      final data = await supabase
          .from('emotion_analysis')
          .select('*')
          .eq('user_id', user.id)
          .order('created_at', ascending: false);

      // Group data by date
      final Map<String, List<Map<String, dynamic>>> groupedData = {};
      
      for (var entry in data) {
        final DateTime createdAt = DateTime.parse(entry['created_at']);
        final String dateKey = DateFormat('yyyy-MM-dd').format(createdAt);
        
        if (!groupedData.containsKey(dateKey)) {
          groupedData[dateKey] = [];
        }
        
        groupedData[dateKey]!.add(entry);
      }

      setState(() {
        _emotionDataByDate = groupedData;
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load emotion data: $e';
        _isLoading = false;
      });
    }
  }

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

  Color _getEmotionColor(String emotion) {
    switch (emotion.toLowerCase()) {
      case 'happy':
      case 'joy':
        return Colors.yellow[700]!;
      case 'sad':
      case 'sadness':
        return Colors.blue[300]!;
      case 'angry':
      case 'anger':
        return Colors.red[400]!;
      case 'fear':
        return Colors.purple[300]!;
      case 'surprise':
        return Colors.orange[300]!;
      case 'disgust':
        return Colors.green[400]!;
      case 'neutral':
        return Colors.grey[400]!;
      default:
        return Colors.grey[500]!;
    }
  }

  Widget _buildEmotionCard(Map<String, dynamic> entry) {
    final DateTime createdAt = DateTime.parse(entry['created_at']);
    final String time = DateFormat('hh:mm a').format(createdAt);
    final String emotion = entry['overall_emotion'];
    final double confidence = (entry['overall_confidence'] as num).toDouble();
    
    // Create a unique ID for this card
    final String cardId = entry['id'].toString();
    final bool isExpanded = _expandedCards.contains(cardId);
    
    return Card(
      elevation: 3,
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 4),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: InkWell(
        onTap: () {
          setState(() {
            if (isExpanded) {
              _expandedCards.remove(cardId);
            } else {
              _expandedCards.add(cardId);
            }
          });
        },
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    time,
                    style: const TextStyle(
                      fontWeight: FontWeight.w500,
                      fontSize: 14,
                    ),
                  ),
                  Row(
                    children: [
                      Text(
                        '${(confidence * 100).toStringAsFixed(0)}%',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Icon(
                        isExpanded ? Icons.keyboard_arrow_up : Icons.keyboard_arrow_down,
                        size: 16,
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      color: _getEmotionColor(emotion),
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    emotion.toUpperCase(),
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                    ),
                  ),
                ],
              ),
              if (isExpanded && entry['questions'] != null) ...[
                const SizedBox(height: 16),
                const Divider(),
                const SizedBox(height: 8),
                _buildQuestionDetails(entry['questions']),
              ],
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildQuestionDetails(dynamic questionsData) {
    // Handle different formats of questions data
    if (questionsData is List) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Assessment Details:',
            style: TextStyle(
              fontWeight: FontWeight.w500,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          ...questionsData.map((question) {
            if (question is Map) {
              // Extract relevant information from the question
              final String? questionText = question['text']?.toString() ?? question['question']?.toString();
              final String? answer = question['answer']?.toString();
              final String? emotion = question['emotion']?.toString();
              
              if (questionText != null) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Q: $questionText',
                        style: const TextStyle(
                          fontWeight: FontWeight.w500,
                          fontSize: 13,
                        ),
                      ),
                      if (answer != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 4, left: 8),
                          child: Text(
                            'A: $answer',
                            style: const TextStyle(fontSize: 13),
                          ),
                        ),
                      if (emotion != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 4, left: 8),
                          child: Row(
                            children: [
                              Container(
                                width: 12,
                                height: 12,
                                decoration: BoxDecoration(
                                  color: _getEmotionColor(emotion),
                                  shape: BoxShape.circle,
                                ),
                              ),
                              const SizedBox(width: 6),
                              Text(
                                emotion.toUpperCase(),
                                style: const TextStyle(
                                  fontSize: 12,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                );
              }
            }
            // If it's just a string (like a transcript)
            if (question is String && question.trim().isNotEmpty) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  question,
                  style: const TextStyle(fontSize: 13),
                ),
              );
            }
            return const SizedBox.shrink();
          }).toList(),
        ],
      );
    } else if (questionsData is Map) {
      // If questions is a map, display its key-value pairs
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Assessment Details:',
            style: TextStyle(
              fontWeight: FontWeight.w500,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          ...questionsData.entries.map((entry) {
            final key = entry.key;
            final value = entry.value;
            
            if (value is String && value.trim().isNotEmpty) {
              return Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      key,
                      style: const TextStyle(
                        fontWeight: FontWeight.w500,
                        fontSize: 13,
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(left: 8, top: 4),
                      child: Text(
                        value,
                        style: const TextStyle(fontSize: 13),
                      ),
                    ),
                  ],
                ),
              );
            }
            return const SizedBox.shrink();
          }).toList(),
        ],
      );
    } else if (questionsData is String && questionsData.trim().isNotEmpty) {
      // If questions is just a string, display it directly
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Transcript:',
            style: TextStyle(
              fontWeight: FontWeight.w500,
              fontSize: 14,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            questionsData,
            style: const TextStyle(fontSize: 13),
          ),
        ],
      );
    }
    
    // Default case if we can't parse the data
    return const Text(
      'No detailed information available',
      style: TextStyle(
        fontStyle: FontStyle.italic,
        color: Colors.grey,
        fontSize: 13,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Emotional Stats'),
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
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _fetchEmotionData,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(Icons.error_outline, size: 48, color: Colors.red),
                      const SizedBox(height: 16),
                      Text(_error!, textAlign: TextAlign.center),
                      const SizedBox(height: 24),
                      ElevatedButton(
                        onPressed: _fetchEmotionData,
                        child: const Text('Try Again'),
                      ),
                    ],
                  ),
                )
              : _emotionDataByDate.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.sentiment_neutral, size: 64, color: Colors.grey),
                          const SizedBox(height: 16),
                          const Text(
                            'No emotion data found',
                            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 8),
                          const Text(
                            'Complete an assessment to see your stats',
                            style: TextStyle(fontSize: 14, color: Colors.grey),
                          ),
                          const SizedBox(height: 24),
                          ElevatedButton(
                            onPressed: () => _onNavTap(1),  // Navigate to assessment page
                            child: const Text('Start Assessment'),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.all(16),
                      itemCount: _emotionDataByDate.length,
                      itemBuilder: (context, index) {
                        final String date = _emotionDataByDate.keys.elementAt(index);
                        final List<Map<String, dynamic>> entries = _emotionDataByDate[date]!;
                        final DateTime dateTime = DateTime.parse(date);
                        final String formattedDate = DateFormat('EEEE, MMMM d, yyyy').format(dateTime);
                        
                        return Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            if (index > 0) const SizedBox(height: 16),
                            Container(
                              padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                              decoration: BoxDecoration(
                                color: isDark ? Colors.green[900]!.withOpacity(0.4) : Colors.green[50],
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                formattedDate,
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                  color: isDark ? Colors.white : Colors.green[900],
                                ),
                              ),
                            ),
                            const SizedBox(height: 8),
                            ...entries.map((entry) => _buildEmotionCard(entry)).toList(),
                          ],
                        );
                      },
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