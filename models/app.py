import os
from datetime import datetime
from io import BytesIO
from collections import Counter
import json
import pickle

import numpy as np
from PIL import Image
import librosa
import joblib
import tensorflow as tf
from supabase import create_client
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import whisper
from typing import List, Dict, Any, Optional
import uvicorn
import glob
import torch

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI app
app = FastAPI()

# Model initialization checker
def ensure_models_loaded():
    """Ensure all models are loaded correctly"""
    global face_model, audio_model, text_model, tokenizer, dataset, whisper_model
    
    models_status = {
        "face_model": False,
        "audio_model": False,
        "text_model": False,
        "whisper_model": False
    }
    
    # Check face model
    try:
        if 'face_model' in globals() and face_model is not None:
            models_status["face_model"] = True
        else:
            print("Face model not properly loaded, trying to load...")
            # Try to find the model with _100.h5 suffix
            face_model_path = "emotion_recognition_model_100.h5"
            if os.path.exists(face_model_path):
                print(f"Found face model at: {face_model_path}")
                face_model = tf.keras.models.load_model(face_model_path)
                models_status["face_model"] = True
            else:
                print(f"Face model not found at: {face_model_path}")
    except Exception as e:
        print(f"Error checking face model: {e}")
    
    # Check audio model
    try:
        if 'audio_model' in globals() and audio_model is not None:
            models_status["audio_model"] = True
        else:
            print("Audio model not properly loaded, trying to load...")
            audio_model_dir = "models/Speech Emotion Recognition Minor Project/LSTM Model"
            audio_model_file = os.path.join(audio_model_dir, "speech_emotion_recognition_model_lstm.h5")
            if os.path.exists(audio_model_file):
                print(f"Found audio model at: {audio_model_file}")
                audio_model = tf.keras.models.load_model(audio_model_file)
                models_status["audio_model"] = True
            else:
                print(f"Audio model not found at: {audio_model_file}")
    except Exception as e:
        print(f"Error checking audio model: {e}")
    
    # Check text model
    try:
        if 'text_model' in globals() and 'tokenizer' in globals() and text_model is not None and tokenizer is not None:
            models_status["text_model"] = True
        else:
            print("Text model not properly loaded, trying to load...")
            text_model_path = "models/emotion-distilbert-model"
            
            if os.path.exists(text_model_path):
                print(f"Found text model at: {text_model_path}")
                from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
                from datasets import load_dataset
                
                tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_path)
                text_model = DistilBertForSequenceClassification.from_pretrained(text_model_path)
                dataset = load_dataset("dair-ai/emotion")
                models_status["text_model"] = True
            else:
                print(f"Text model directory not found: {text_model_path}")
    except Exception as e:
        print(f"Error checking text model: {e}")
    
    # Check whisper model
    try:
        if 'whisper_model' in globals() and whisper_model is not None:
            models_status["whisper_model"] = True
        else:
            print("Whisper model not properly loaded")
            # Try to load whisper model
            try:
                whisper_model = whisper.load_model("base")
                print("Successfully loaded Whisper model")
                models_status["whisper_model"] = True
            except Exception as e:
                print(f"Error loading Whisper model: {e}")
    except Exception as e:
        print(f"Error checking whisper model: {e}")
    
    return models_status

@app.get("/test_models")
async def test_models():
    """Test endpoint to check if all models are loaded correctly"""
    models_status = ensure_models_loaded()
    all_loaded = all(models_status.values())
    
    return {
        "success": all_loaded,
        "models_status": models_status,
        "message": "All models loaded successfully" if all_loaded else "Some models failed to load"
    }

# Load Models
try:
    # Face Emotion Model - using direct file path
    face_model_path = "models/emotion_recognition_model_100.h5"
    face_model = tf.keras.models.load_model(face_model_path)
    print(f"Face model loaded from: {face_model_path}")
    
    # Audio Emotion Model - using the approach from TesterLSTM.py
    audio_model_dir = "models/Speech Emotion Recognition Minor Project/LSTM Model"
    audio_model_file = os.path.join(audio_model_dir, "speech_emotion_recognition_model_lstm.h5")
    audio_model = tf.keras.models.load_model(audio_model_file)
    print(f"Audio model loaded from: {audio_model_file}")
    
    # Text Emotion Model - using the approach from test.py
    text_model_path = "models/emotion-distilbert-model"
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    from datasets import load_dataset
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(text_model_path)
    text_model = DistilBertForSequenceClassification.from_pretrained(text_model_path)
    dataset = load_dataset("dair-ai/emotion")
    print(f"Text model loaded from: {text_model_path}")
    
    # Whisper for speech-to-text
    whisper_model = whisper.load_model("base")
    print("Whisper model loaded successfully")
    
    # Load scaler for audio model
    scaler_path = "models/Speech Emotion Recognition Minor Project/LSTM Model/scalerLSTM.pkl"
    scaler = pickle.load(open(scaler_path, 'rb'))
    
    print("All models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Continuing with limited functionality")

# Define the standard questions
STANDARD_QUESTIONS = [
    "How are you feeling today?",
    "Can you describe your current mood?",
    "Have you experienced feelings of anxiety recently?"
]

# ========== Supabase Utilities ==========

def download_file(bucket: str, path: str) -> bytes:
    """Download a file from Supabase storage"""
    try:
        data = supabase.storage.from_(bucket).download(path)
        if data is None:
            raise FileNotFoundError(f"File not found: {path}")
        return data
    except Exception as e:
        print(f"Error downloading {path} from {bucket}: {str(e)}")
        raise

def list_folder_contents(bucket: str, folder_path: str) -> list:
    """List files in a folder"""
    try:
        return supabase.storage.from_(bucket).list(folder_path)
    except Exception as e:
        print(f"Error listing contents of {folder_path} in {bucket}: {str(e)}")
        return []

# ========== Image (Face) Prediction ==========

def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Convert image bytes to model input format"""
    try:
        # Convert to grayscale by using 'L' mode
        img = Image.open(BytesIO(img_bytes)).convert('L')
        img = img.resize((48, 48))
        img_array = np.array(img) / 255.0
        # Add channel dimension at the end
        img_array = np.expand_dims(img_array, axis=-1)
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        # Return a blank image if there's an error - with correct shape
        return np.zeros((1, 48, 48, 1))

def predict_face_emotion(image_bytes: bytes) -> str:
    """Predict emotion from a single facial image"""
    try:
        preprocessed = preprocess_image(image_bytes)
        preds = face_model.predict(preprocessed, verbose=0)
        emotion_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        emotion_idx = np.argmax(preds[0])
        return emotion_labels[emotion_idx]
    except Exception as e:
        print(f"Error predicting face emotion: {str(e)}")
        return "unknown"

def predict_max_face_emotion(images: list) -> dict:
    """Predict emotion from multiple facial images and get the most common one"""
    if not images:
        return {"emotion": "unknown", "confidence": 0.0}
    
    try:
        predictions = [predict_face_emotion(img) for img in images]
        emotion_counts = Counter(predictions)
        most_common = emotion_counts.most_common(1)[0]
        
        # Calculate confidence as percentage of images with that emotion
        emotion = most_common[0]
        count = most_common[1]
        confidence = count / len(predictions)
        
        return {
            "emotion": emotion, 
            "confidence": confidence,
            "all_emotions": {e: c/len(predictions) for e, c in emotion_counts.items()}
        }
    except Exception as e:
        print(f"Error in face emotion aggregation: {str(e)}")
        return {"emotion": "error", "confidence": 0.0}

# ========== Audio Emotion ==========

def extract_features(data, sample_rate):
    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    # RMS Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    return np.hstack((zcr, chroma_stft, mfcc, rms, mel))

def extract_audio_features(audio_bytes: bytes) -> np.ndarray:
    """Extract features from audio for model input (TesterLSTM.py style)"""
    try:
        # Write bytes to temp file
        aac_path = "temp_audio_original.aac"
        wav_path = "temp_audio.wav"
        
        with open(aac_path, "wb") as f:
            f.write(audio_bytes)
            
        # Convert AAC to WAV using FFmpeg if available
        try:
            import subprocess
            subprocess.call(['ffmpeg', '-i', aac_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y'])
            print(f"Successfully converted audio to WAV format")
        except Exception as e:
            print(f"Error converting audio format: {e}, using original file")
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
                
        # Load the audio file
        data, sample_rate = librosa.load(wav_path, duration=2.5, offset=0.6)
        # Extract features
        features = extract_features(data, sample_rate)
        # Scale the features using the loaded scaler
        features_scaled = scaler.transform(features.reshape(1, -1))
        # Reshape for the model
        features_reshaped = np.expand_dims(features_scaled, axis=2)  # (1, 162, 1)
        # Clean up
        try:
            os.remove(aac_path)
            os.remove(wav_path)
        except:
            pass
        return features_reshaped
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty features with correct shape
        return np.zeros((1, 162, 1))

def predict_audio_emotion(audio_bytes: bytes) -> dict:
    """Predict emotion from audio"""
    try:
        features = extract_audio_features(audio_bytes)
        preds = audio_model.predict(features, verbose=0)
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        emotion_idx = np.argmax(preds[0])
        confidence = float(preds[0][emotion_idx])
        return {
            "emotion": emotions[emotion_idx],
            "confidence": confidence,
            "all_emotions": {emotions[i]: float(preds[0][i]) for i in range(len(emotions))}
        }
    except Exception as e:
        print(f"Error predicting audio emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"emotion": "unknown", "confidence": 0.0}

# ========== Text Emotion ==========

def audio_to_text(audio_bytes: bytes) -> str:
    """Convert audio to text using Whisper"""
    try:
        # Check if whisper_model is defined in global scope
        global whisper_model
        if 'whisper_model' not in globals():
            print("Whisper model not loaded globally, loading now...")
            whisper_model = whisper.load_model("base")
        
        # Write bytes to temp file
        aac_path = "temp_whisper_original.aac"
        wav_path = "temp_whisper.wav"
        
        with open(aac_path, "wb") as f:
            f.write(audio_bytes)
            
        # Convert AAC to WAV using FFmpeg if available
        try:
            import subprocess
            subprocess.call(['ffmpeg', '-i', aac_path, '-acodec', 'pcm_s16le', '-ar', '16000', wav_path, '-y'])
            print(f"Successfully converted audio to WAV format for whisper")
        except Exception as e:
            print(f"Error converting audio format for whisper: {e}, using original file")
            with open(wav_path, "wb") as f:
                f.write(audio_bytes)
        
        # Verify the file exists and has content
        if not os.path.exists(wav_path):
            print(f"Error: Temp file {wav_path} was not created")
            return ""
        
        file_size = os.path.getsize(wav_path)
        print(f"Temp audio file created: {wav_path}, size: {file_size} bytes")
        
        if file_size == 0:
            print("Error: Audio file is empty (0 bytes)")
            return ""
        
        # Use absolute path
        abs_path = os.path.abspath(wav_path)
        print(f"Using absolute path for transcription: {abs_path}")
        
        # Transcribe
        result = whisper_model.transcribe(abs_path)
        
        # Clean up
        try:
            os.remove(aac_path)
            os.remove(wav_path)
            print(f"Temp files removed")
        except Exception as e:
            print(f"Warning: Could not remove temp files: {e}")
        
        return result['text']
    except Exception as e:
        print(f"Error transcribing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return ""

def predict_text_emotion(text: str) -> dict:
    """Predict emotion from text using DistilBERT"""
    if not text.strip():
        return {"emotion": "unknown", "confidence": 0.0}
    
    try:
        # Approach from test.py
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        text_model.eval()
        with torch.no_grad():
            outputs = text_model(**inputs)
        
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_idx = torch.argmax(logits, dim=1).item()
        confidence = float(probs[0][pred_idx])
        emotion = dataset['train'].features['label'].int2str(pred_idx)
        
        # Map all probabilities
        all_emotions = {}
        for i in range(probs.shape[1]):
            emotion_name = dataset['train'].features['label'].int2str(i)
            all_emotions[emotion_name] = float(probs[0][i])
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": all_emotions
        }
    except Exception as e:
        print(f"Error predicting text emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"emotion": "unknown", "confidence": 0.0}

# ========== Aggregation ==========

def aggregate_emotions(face_result: dict, audio_result: dict, text_result: dict) -> dict:
    """Aggregate emotions from all modalities"""
    try:
        # Get primary emotions
        face_emotion = face_result.get("emotion", "unknown")
        audio_emotion = audio_result.get("emotion", "unknown")
        text_emotion = text_result.get("emotion", "unknown")
        
        # Get confidences
        face_conf = face_result.get("confidence", 0.0)
        audio_conf = audio_result.get("confidence", 0.0)
        text_conf = text_result.get("confidence", 0.0)
        
        # Give weight based on confidence
        emotions = []
        if face_emotion != "unknown":
            emotions.extend([face_emotion] * int(face_conf * 10))
        if audio_emotion != "unknown":
            emotions.extend([audio_emotion] * int(audio_conf * 10))
        if text_emotion != "unknown":
            emotions.extend([text_emotion] * int(text_conf * 10))
        
        if not emotions:
            return {"emotion": "neutral", "confidence": 0.0}
        
        # Count occurrences
        emotion_counts = Counter(emotions)
        most_common = emotion_counts.most_common(1)[0]
        emotion = most_common[0]
        confidence = most_common[1] / len(emotions)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": {e: c/len(emotions) for e, c in emotion_counts.items()},
            "modalities": {
                "face": face_emotion,
                "audio": audio_emotion,
                "text": text_emotion
            }
        }
    except Exception as e:
        print(f"Error in emotion aggregation: {str(e)}")
        return {"emotion": "neutral", "confidence": 0.0}

def aggregate_all_questions(question_results: List[Dict]) -> dict:
    """Aggregate results from all questions"""
    try:
        all_emotions = []
        for result in question_results:
            emotion = result.get("aggregate", {}).get("emotion")
            if emotion and emotion != "unknown":
                all_emotions.append(emotion)
        
        if not all_emotions:
            return {"emotion": "neutral", "confidence": 0.0}
        
        emotion_counts = Counter(all_emotions)
        most_common = emotion_counts.most_common(1)[0]
        emotion = most_common[0]
        confidence = most_common[1] / len(all_emotions)
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "all_emotions": {e: c/len(all_emotions) for e, c in emotion_counts.items()},
        }
    except Exception as e:
        print(f"Error in question aggregation: {str(e)}")
        return {"emotion": "neutral", "confidence": 0.0}

def find_todays_folders(email: str):
    """Find folders created today for a specific user email"""
    try:
        # List folders for this user
        folders = list_folder_contents("interview-images", email)
        
        # Extract dates from folder names (format: yyyy-MM-dd_question)
        today = datetime.now().strftime("%Y-%m-%d")
        today_folders = []
        all_folders = []
        
        print(f"Looking for folders with today's date: {today}")
        print(f"All folders for user {email}: {[f['name'] for f in folders if 'name' in f]}")
        
        # Skip the JSON lookup and directly check folder names
        for folder in folders:
            folder_name = folder.get('name', '')
            if folder_name:
                all_folders.append(folder_name)
                if today in folder_name:
                    today_folders.append(folder_name)
                    print(f"Found folder matching today's date: {folder_name}")
        
        if not today_folders:
            print(f"No folders found for today ({today}) for user {email}, using most recent folders")
            return all_folders
        else:
            print(f"Found {len(today_folders)} folders for today")
            return today_folders

    except Exception as e:
        print(f"Error finding today's folders: {e}")
        return []
# ========== Additional API Models ==========

class EmotionRequest(BaseModel):
    email: str
    date: Optional[str] = None

class InterviewRequest(BaseModel):
    emailid: str
    uid: str
    date: Optional[str] = None

# ========== API Endpoints ==========

@app.post("/face_predict/")
async def face_predict(data: EmotionRequest):
    """Process facial emotions for all questions of a user from today's images"""
    try:
        # Ensure models are loaded
        models_status = ensure_models_loaded()
        if not models_status["face_model"]:
            return {"success": False, "error": "Face emotion model not loaded"}
        
        # Use today's date if not provided
        date_str = data.date if data.date else datetime.now().strftime("%Y-%m-%d")
        print(f"Using date: {date_str}")
        
        # Look for actual folders from today
        today_folders = find_todays_folders(data.email)
        
        question_results = []
        
        # Process each question
        for i, question in enumerate(STANDARD_QUESTIONS):
            try:
                # Find folders for this question
                sanitized_question = question.replace(" ", "_").replace("?", "")
                matching_folders = [folder for folder in today_folders if sanitized_question in folder]
                
                images = []
                
                if matching_folders:
                    print(f"Found matching folders for question '{question}': {matching_folders}")
                    # Use the most recent folder
                    full_folder_path = f"{data.email}/{matching_folders[-1]}"
                    
                    # List images in the folder
                    images_list = list_folder_contents("interview-images", full_folder_path)
                    
                    # Download each image
                    for img in images_list:
                        if not img['name'].endswith(('.jpg', '.jpeg', '.png')):
                            continue
                        path = f"{full_folder_path}/{img['name']}"
                        try:
                            img_bytes = download_file("interview-images", path)
                            images.append(img_bytes)
                        except Exception as e:
                            print(f"Error downloading image {path}: {str(e)}")
                else:
                    print(f"No matching folders found for question: {question}")
                
                # Process face emotions for this question's images
                if images:
                    face_result = predict_max_face_emotion(images)
                    question_results.append({
                        "question": question,
                        "question_index": i,
                        "face_emotion": face_result["emotion"],
                        "confidence": face_result["confidence"],
                        "all_emotions": face_result.get("all_emotions", {})
                    })
                else:
                    question_results.append({
                        "question": question,
                        "question_index": i,
                        "face_emotion": "unknown",
                        "confidence": 0.0,
                        "error": "No images found for this question"
                    })
                
            except Exception as e:
                print(f"Error processing face emotions for question '{question}': {str(e)}")
                question_results.append({
                    "question": question,
                    "question_index": i,
                    "face_emotion": "error",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "date": date_str,
            "results": question_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/audio_predict/")
async def audio_predict(data: EmotionRequest):
    """Process audio emotions and transcripts for all questions of a user from today's audio files"""
    try:
        # Ensure models are loaded
        models_status = ensure_models_loaded()
        if not models_status["audio_model"]:
            return {"success": False, "error": "Audio emotion model not loaded"}
        
        # Use today's date if not provided
        date_str = data.date if data.date else datetime.now().strftime("%Y-%m-%d")
        print(f"Using date: {date_str}")
        
        # Look for actual folders from today
        today_folders = find_todays_folders(data.email)
        
        question_results = []
        
        # Process each question
        for i, question in enumerate(STANDARD_QUESTIONS):
            try:
                # Find folders for this question
                sanitized_question = question.replace(" ", "_").replace("?", "")
                matching_folders = [folder for folder in today_folders if sanitized_question in folder]
                
                audio_bytes = None
                
                if matching_folders:
                    print(f"Found matching folders for question '{question}': {matching_folders}")
                    # Get folder name from the matched folder
                    folder_name = matching_folders[-1]
                    
                    # Direct path: userEmail/date_sanitizedQuestion.aac
                    audio_path = f"{data.email}/{folder_name}.aac"
                    try:
                        audio_bytes = download_file("interview-audio", audio_path)
                        print(f"Found and downloaded audio file directly: {audio_path}")
                    except Exception as e:
                        print(f"Error downloading direct audio path: {str(e)}")
                        audio_bytes = None
                        
                    # If direct approach failed, try old method with folder listing
                    if audio_bytes is None:
                        print("Trying fallback method to find audio for text...")
                        full_folder_path = f"{data.email}"
                        
                        # List files in the folder
                        audio_files = list_folder_contents("interview-audio", full_folder_path)
                        
                        # Try to find audio file with various extensions
                        for audio_file in audio_files:
                            file_name = audio_file.get('name', '')
                            if folder_name in file_name and file_name.endswith(('.wav', '.aac', '.mp3', '.m4a')):
                                try:
                                    audio_path = f"{full_folder_path}/{file_name}"
                                    audio_bytes = download_file("interview-audio", audio_path)
                                    print(f"Found and downloaded audio file: {audio_path}")
                                    break
                                except Exception as e:
                                    print(f"Error downloading audio {audio_path}: {str(e)}")
                else:
                    print(f"No matching folders found for question: {question}")
                
                # Process audio emotions and transcript for this question
                result = {
                    "question": question,
                    "question_index": i
                }
                
                if audio_bytes:
                    # Audio emotion
                    audio_result = predict_audio_emotion(audio_bytes)
                    result["audio_emotion"] = audio_result["emotion"]
                    result["audio_confidence"] = audio_result["confidence"]
                    result["all_audio_emotions"] = audio_result.get("all_emotions", {})
                    
                    # Speech to text
                    transcript = audio_to_text(audio_bytes)
                    result["transcript"] = transcript
                else:
                    result["audio_emotion"] = "unknown"
                    result["audio_confidence"] = 0.0
                    result["transcript"] = ""
                    result["error"] = "No audio found for this question"
                
                question_results.append(result)
                
            except Exception as e:
                print(f"Error processing audio for question '{question}': {str(e)}")
                question_results.append({
                    "question": question,
                    "question_index": i,
                    "audio_emotion": "error",
                    "transcript": "",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "date": date_str,
            "results": question_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/text_predict/")
async def text_predict(data: EmotionRequest):
    """Process text emotions from audio transcripts for all questions of a user"""
    try:
        # Ensure models are loaded
        models_status = ensure_models_loaded()
        if not models_status["text_model"] or not models_status["whisper_model"]:
            return {"success": False, "error": "Text emotion model or Whisper model not loaded"}
        
        # Use today's date if not provided
        date_str = data.date if data.date else datetime.now().strftime("%Y-%m-%d")
        print(f"Using date: {date_str}")
        
        # Look for actual folders from today
        today_folders = find_todays_folders(data.email)
        
        question_results = []
        
        # Process each question
        for i, question in enumerate(STANDARD_QUESTIONS):
            try:
                # Find folders for this question
                sanitized_question = question.replace(" ", "_").replace("?", "")
                matching_folders = [folder for folder in today_folders if sanitized_question in folder]
                
                audio_bytes = None
                
                if matching_folders:
                    print(f"Found matching folders for question '{question}': {matching_folders}")
                    # Get folder name from the matched folder
                    folder_name = matching_folders[-1]
                    
                    # Direct path: userEmail/date_sanitizedQuestion.aac
                    audio_path = f"{data.email}/{folder_name}.aac"
                    try:
                        audio_bytes = download_file("interview-audio", audio_path)
                        print(f"Found and downloaded audio file directly: {audio_path}")
                    except Exception as e:
                        print(f"Error downloading direct audio path: {str(e)}")
                        audio_bytes = None
                        
                    # If direct approach failed, try old method with folder listing
                    if audio_bytes is None:
                        print("Trying fallback method to find audio for text...")
                        full_folder_path = f"{data.email}"
                        
                        # List files in the folder
                        audio_files = list_folder_contents("interview-audio", full_folder_path)
                        
                        # Try to find audio file with various extensions
                        for audio_file in audio_files:
                            file_name = audio_file.get('name', '')
                            if folder_name in file_name and file_name.endswith(('.wav', '.aac', '.mp3', '.m4a')):
                                try:
                                    audio_path = f"{full_folder_path}/{file_name}"
                                    audio_bytes = download_file("interview-audio", audio_path)
                                    print(f"Found and downloaded audio file: {audio_path}")
                                    break
                                except Exception as e:
                                    print(f"Error downloading audio {audio_path}: {str(e)}")
                else:
                    print(f"No matching folders found for question: {question}")
                
                # Process text emotions from transcript for this question
                result = {
                    "question": question,
                    "question_index": i
                }
                
                if audio_bytes:
                    # Speech to text
                    transcript = audio_to_text(audio_bytes)
                    result["transcript"] = transcript
                    
                    # Text emotion analysis
                    if transcript:
                        text_result = predict_text_emotion(transcript)
                        result["text_emotion"] = text_result["emotion"]
                        result["text_confidence"] = text_result["confidence"]
                        result["all_text_emotions"] = text_result.get("all_emotions", {})
                    else:
                        result["text_emotion"] = "unknown"
                        result["text_confidence"] = 0.0
                        result["error"] = "No transcript could be generated"
                else:
                    result["transcript"] = ""
                    result["text_emotion"] = "unknown"
                    result["text_confidence"] = 0.0
                    result["error"] = "No audio found for this question"
                
                question_results.append(result)
                
            except Exception as e:
                print(f"Error processing text for question '{question}': {str(e)}")
                question_results.append({
                    "question": question,
                    "question_index": i,
                    "text_emotion": "error",
                    "transcript": "",
                    "error": str(e)
                })
        
        return {
            "success": True,
            "date": date_str,
            "results": question_results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Keep the old endpoint for backward compatibility but simplify it
@app.post("/process_all_questions/")
async def process_all_questions(data: InterviewRequest):
    """Process all three questions for a user by combining all modalities"""
    try:
        # Convert to the new format
        emotion_request = EmotionRequest(email=data.emailid, date=data.date)
        
        # Get results from each modality
        face_response = await face_predict(emotion_request)
        audio_response = await audio_predict(emotion_request)
        text_response = await text_predict(emotion_request)
        
        if not face_response["success"] or not audio_response["success"] or not text_response["success"]:
            return {
                "success": False,
                "error": "One or more prediction services failed"
            }
        
        # Prepare the combined results
        all_results = []
        date_str = face_response["date"]  # All should have the same date
        
        # For each question, combine the results
        for i, question in enumerate(STANDARD_QUESTIONS):
            try:
                face_result = next((q for q in face_response["results"] if q["question_index"] == i), {})
                audio_result = next((q for q in audio_response["results"] if q["question_index"] == i), {})
                text_result = next((q for q in text_response["results"] if q["question_index"] == i), {})
                
                question_result = {
                    "question": question,
                    "question_index": i,
                    "face": {
                        "emotion": face_result.get("face_emotion", "unknown"),
                        "confidence": face_result.get("confidence", 0.0)
                    },
                    "audio": {
                        "emotion": audio_result.get("audio_emotion", "unknown"), 
                        "confidence": audio_result.get("audio_confidence", 0.0)
                    },
                    "text": {
                        "emotion": text_result.get("text_emotion", "unknown"),
                        "confidence": text_result.get("text_confidence", 0.0)
                    },
                    "transcript": text_result.get("transcript", ""),
                    "aggregate": {}
                }
                
                # Aggregate emotions for this question
                aggregate = aggregate_emotions(
                    question_result["face"],
                    question_result["audio"],
                    question_result["text"]
                )
                question_result["aggregate"] = aggregate
                
                all_results.append(question_result)
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                all_results.append({
                    "question": question,
                    "question_index": i,
                    "error": str(e)
                })
        
        # Get overall emotion across all questions
        overall_result = aggregate_all_questions(all_results)
        print(overall_result)
        
        return {
            "success": True,
            "date": date_str,
            "questions": all_results,
            "overall": overall_result
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
