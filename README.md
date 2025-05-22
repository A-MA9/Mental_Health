<p align="center">
  <img src="assets/logo.png" alt="Manan Logo" width="120" />
</p>

<h1 align="center">मनन (Manan) - Mental Health Companion App</h1>

<p align="center">
  <b>AI-powered, privacy-first mental health companion app built with Flutter, Supabase, and Machine Learning.</b>
</p>

<p align="center">
  <a href="https://flutter.dev/"><img src="https://img.shields.io/badge/Flutter-3.7-blue?logo=flutter" alt="Flutter"></a>
  <a href="https://supabase.com/"><img src="https://img.shields.io/badge/Supabase-Backend-green?logo=supabase" alt="Supabase"></a>
  <a href="https://github.com/your-username/your-repo-name/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Platform-Android%20%7C%20iOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" alt="Status">
</p>

---

## ✨ Features

- **Emotion Analysis**: Multimodal emotion detection using face, voice, and text.
- **Animated Pie Chart Visualization**: See your daily emotional stats with beautiful, animated charts.
- **AI Chatbot**: Friendly AI companion to chat and support you.
- **Video/Audio Assessment**: Record and analyze your mood using your camera and microphone.
- **Profile & Authentication**: Secure sign-up, login, and profile management (Supabase Auth).
- **Deep Linking**: Seamless authentication and verification via email links.
- **Cross-Platform**: Works on both Android and iOS.
- **Modern UI**: Beautiful, accessible, and responsive Flutter design.
- **Privacy-first**: Your data is securely stored and never shared.

---

## 🛠️ Tech Stack

- **Flutter** (Dart)
- **Supabase** (Postgres, Auth, Storage)
- **Python** (Machine Learning models)
- **TensorFlow, PyTorch, Transformers** (for emotion models)
- **fl_chart** (for animated charts)
- **Other Flutter packages**: `google_fonts`, `curved_navigation_bar`, `camera`, `flutter_sound`, `permission_handler`, `audioplayers`, `app_links`, `flutter_launcher_icons`, and more.

---

## 📱 Screenshots

<p align="center">
  <img src="assets/logo.png" alt="App Screenshot" width="200"/>
  <!-- Add more screenshots here if available -->
</p>

---

## 🚀 Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

```sh
flutter pub get
```

### 3. Set up Supabase

- Follow the instructions in [SUPABASE_SETUP.md](SUPABASE_SETUP.md) to set up your backend and environment variables.
- For deep linking, see [DEEP_LINK_GUIDE.md](DEEP_LINK_GUIDE.md).

### 4. Run the app

```sh
flutter run
```

### 5. Build for release

```sh
flutter build apk --release
# or for Play Store
flutter build appbundle --release
```

---

## ⚙️ Project Structure

- `lib/` - Main Flutter app code (UI, logic, screens)
- `models/` - Python scripts and ML models for emotion analysis
- `assets/` - Images, icons, and other assets
- `supabase/` - Backend functions, SQL migrations, and config
- `test/` - Flutter widget and integration tests

---

## 🧠 How It Works

- **Emotion Analysis**: Uses ML models (face, audio, text) to analyze your mood.
- **Stats & Visualization**: Animated pie charts show your daily emotional breakdown.
- **Chatbot & Assessment**: Interact with the AI or take a guided video/audio assessment.
- **Data Storage**: All data is securely stored in Supabase with row-level security.

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Flutter](https://flutter.dev/)
- [Supabase](https://supabase.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [fl_chart](https://pub.dev/packages/fl_chart)
- All open-source contributors and the mental health community.

---

> **Made with ❤️ for your mental well-being.**
