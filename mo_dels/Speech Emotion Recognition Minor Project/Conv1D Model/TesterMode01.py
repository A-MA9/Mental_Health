# -*- coding: utf-8 -*-
"""Test Speech Emotion Recognition Model

This script loads a trained Keras model (in H5 format) and tests it
on audio files located in a specified folder. It extracts features
from each audio file and predicts the emotion using the loaded model.
"""

import os
import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from keras.models import load_model

# --- Configuration ---
model_path = "speech_emotion_recognition_model.h5"
test_audio_folder = "random_audio_samples_by_emotion"
scaler_path = "scalerModel01.pkl"  # You need to save your scaler
encoder_path = "encoderModel01.pkl"  # You need to save your encoder
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load the trained model
try:
    loaded_model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()

# Load the saved scaler
try:
    scaler = pickle.load(open(scaler_path, 'rb'))
    print(f"Scaler loaded successfully from {scaler_path}")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {scaler_path}. "
          "Make sure you saved the scaler during training.")
    exit()

# Load the saved OneHotEncoder (though not directly used for prediction here)
try:
    encoder = pickle.load(open(encoder_path, 'rb'))
    print(f"Encoder loaded successfully from {encoder_path}")
except FileNotFoundError:
    print(f"Error: Encoder file not found at {encoder_path}. "
          "Make sure you saved the encoder during training.")
    exit()

# Function to extract features (must be the same as used during training)
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

# Initialize lists to store results
predictions = []
audio_file_names = []
actual_emotions = []  # To store the actual emotion from the filename (if available)

# Iterate through all files in the test audio folder
for filename in os.listdir(test_audio_folder):
    if filename.endswith(('.wav', '.mp3')):  # Add other audio file extensions if needed
        file_path = os.path.join(test_audio_folder, filename)
        try:
            # Load the audio file
            audio, sample_rate = librosa.load(file_path, duration=2.5, offset=0.6) # Same duration and offset as training

            # Extract features
            features = extract_features(audio, sample_rate)

            # Scale the features using the loaded scaler
            features_scaled = scaler.transform(features.reshape(1, -1))

            # Reshape for the model
            features_reshaped = np.expand_dims(features_scaled, axis=2)

            # Make prediction
            predicted_probabilities = loaded_model.predict(features_reshaped)
            predicted_emotion_index = np.argmax(predicted_probabilities)
            predicted_emotion = emotions[predicted_emotion_index]

            predictions.append(predicted_emotion)
            audio_file_names.append(filename)

            # Try to extract the actual emotion from the filename (you might need to adjust this)
            try:
                actual_emotion = filename.split('_')[0].lower() # Assuming filename starts with emotion_...
                if actual_emotion in emotions:
                    actual_emotions.append(actual_emotion)
                else:
                    actual_emotions.append('unknown')
            except:
                actual_emotions.append('unknown')

            print(f"File: {filename}, Predicted Emotion: {predicted_emotion}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

# Create a Pandas DataFrame to display the results
results_df = pd.DataFrame({'Audio File': audio_file_names, 'Predicted Emotion': predictions, 'Actual Emotion': actual_emotions})
print("\n--- Prediction Results ---")
print(results_df)

results_df.to_csv('TestResultModel01.csv', index=False)
print("\nTest Result has been saved to 'TestResultModel01.csv'")

# --- Evaluate Predictions ---
correct_predictions = 0
total_predictions = len(predictions)

if actual_emotions and len(actual_emotions) == total_predictions:
    for predicted, actual in zip(predictions, actual_emotions):
        if predicted == actual:
            correct_predictions += 1

    accuracy = (correct_predictions / total_predictions) * 100
    misclassified = total_predictions - correct_predictions

    print(f"\n--- Evaluation ---")
    print(f"Total Predictions: {total_predictions}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Misclassified Predictions: {misclassified}")
    print(f"Accuracy: {accuracy:.2f}%")
else:
    print("\n--- Evaluation ---")
    print("Actual emotions not available for all files, cannot calculate accuracy.")