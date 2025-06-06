import os
import sys
import pandas as pd  # Import pandas
import numpy as np
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# to play the audio files
from IPython.display import Audio

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define the paths to your datasets
Ravdess = "REVDESS/"  # Replace with your actual path
Crema = "CREMAD/AudioWAV/"  # Replace with your actual path
Tess = "TESS Toronto emotional speech set data/"  # Replace with your actual path
Savee = "SURREY/ALL/"  # Replace with your actual path

# --- RAVDESS Dataset ---
ravdess_directory_list = os.listdir(Ravdess)
file_emotion_ravdess = []
file_path_ravdess = []
for dir in ravdess_directory_list:
    actor = os.listdir(os.path.join(Ravdess, dir))
    for file in actor:
        part = file.split('.')[0].split('-')
        file_emotion_ravdess.append(int(part[2]))
        file_path_ravdess.append(os.path.join(Ravdess, dir, file))

Ravdess_df = pd.DataFrame({'Emotions': file_emotion_ravdess, 'Path': file_path_ravdess})
Ravdess_df['Emotions'].replace({1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'}, inplace=True)
print("RAVDESS DataFrame created.")
print(Ravdess_df.head())

# --- CREMA-D Dataset ---
crema_directory_list = os.listdir(Crema)
file_emotion_crema = []
file_path_crema = []
for file in crema_directory_list:
    file_path_crema.append(os.path.join(Crema, file))
    part = file.split('_')
    if part[2] == 'SAD':
        file_emotion_crema.append('sad')
    elif part[2] == 'ANG':
        file_emotion_crema.append('angry')
    elif part[2] == 'DIS':
        file_emotion_crema.append('disgust')
    elif part[2] == 'FEA':
        file_emotion_crema.append('fear')
    elif part[2] == 'HAP':
        file_emotion_crema.append('happy')
    elif part[2] == 'NEU':
        file_emotion_crema.append('neutral')
    else:
        file_emotion_crema.append('Unknown')

Crema_df = pd.DataFrame({'Emotions': file_emotion_crema, 'Path': file_path_crema})
print("\nCREMA-D DataFrame created.")
print(Crema_df.head())

# --- TESS Dataset ---
tess_directory_list = os.listdir(Tess)
file_emotion_tess = []
file_path_tess = []
for dir in tess_directory_list:
    directories = os.listdir(os.path.join(Tess, dir))
    for file in directories:
        part = file.split('.')[0].split('_')[2]
        if part == 'ps':
            file_emotion_tess.append('surprise')
        else:
            file_emotion_tess.append(part)
        file_path_tess.append(os.path.join(Tess, dir, file))

Tess_df = pd.DataFrame({'Emotions': file_emotion_tess, 'Path': file_path_tess})
print("\nTESS DataFrame created.")
print(Tess_df.head())

# --- SAVEE Dataset ---
savee_directory_list = os.listdir(Savee)
file_emotion_savee = []
file_path_savee = []
for file in savee_directory_list:
    file_path_savee.append(os.path.join(Savee, file))
    ele = file.split('_')[1][:-6]
    if ele == 'a':  # There was a potential typo here, assuming 'ele' should be 'part'
        file_emotion_savee.append('angry')
    elif ele == 'd':
        file_emotion_savee.append('disgust')
    elif ele == 'f':
        file_emotion_savee.append('fear')
    elif ele == 'h':
        file_emotion_savee.append('happy')
    elif ele == 'n':
        file_emotion_savee.append('neutral')
    elif ele == 'sa':
        file_emotion_savee.append('sad')
    else:
        file_emotion_savee.append('surprise')

Savee_df = pd.DataFrame({'Emotions': file_emotion_savee, 'Path': file_path_savee})
print("\nSAVEE DataFrame created.")
print(Savee_df.head())

# --- Concatenate all DataFrames ---
data_path = pd.concat([Ravdess_df, Crema_df, Tess_df, Savee_df], axis=0)

# Save the combined DataFrame to a CSV file
data_path.to_csv("data_path.csv", index=False)

print("\nCombined DataFrame created and saved to data_path.csv")
print(data_path.head())
print(data_path.info())
print(data_path['Emotions'].value_counts())


"""**REMOVING CALM BECAUSE WE DONT NEED IT**"""

data_path = data_path[data_path.Emotions != 'calm']
print(data_path.size)
data_path

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data, sample_rate)  # Pass sample_rate here
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)  # Pass sample_rate here
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)  # Pass sample_rate here
    result = np.vstack((result, res3))  # stacking vertically

    return result

print("Data preprocessing is now begin")

X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

len(X), len(Y), data_path.Path.shape

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features.csv', index=False)
Features.head()

X = Features.iloc[:, :-1].values
Y = Features['labels'].values

# As this is a multiclass classification problem onehotencoding our Y.
encoder = OneHotEncoder()
Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# making our data compatible to model.
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

# --- Save Scaler and Encoder ---
import pickle

# Save the scaler
scaler_filename = 'scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scaler, file)
print(f"Scaler saved as {scaler_filename}")

# Save the OneHotEncoder
encoder_filename = 'encoder.pkl'
with open(encoder_filename, 'wb') as file:
    pickle.dump(encoder, file)
print(f"Encoder saved as {encoder_filename}")

"""## Modelling with LSTM"""

model_lstm = Sequential()
model_lstm.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
model_lstm.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model_lstm.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
model_lstm.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model_lstm.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model_lstm.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model_lstm.add(Dropout(0.2))

model_lstm.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model_lstm.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))

model_lstm.add(LSTM(units=128, return_sequences=False))  # Replacing Flatten with LSTM
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(units=7, activation='softmax'))

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_lstm.summary()

rlrp_lstm = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
history_lstm = model_lstm.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp_lstm])

print("Accuracy of LSTM model on test data:", model_lstm.evaluate(x_test, y_test)[1] * 100, "%")

epochs_lstm = [i for i in range(50)]
fig_lstm, ax_lstm = plt.subplots(1, 2)
train_acc_lstm = history_lstm.history['accuracy']
train_loss_lstm = history_lstm.history['loss']
test_acc_lstm = history_lstm.history['val_accuracy']
test_loss_lstm = history_lstm.history['val_loss']

fig_lstm.set_size_inches(20, 6)
ax_lstm[0].plot(epochs_lstm, train_loss_lstm, label='Training Loss')
ax_lstm[0].plot(epochs_lstm, test_loss_lstm, label='Testing Loss')
ax_lstm[0].set_title('Training & Testing Loss (LSTM)')
ax_lstm[0].legend()
ax_lstm[0].set_xlabel("Epochs")

ax_lstm[1].plot(epochs_lstm, train_acc_lstm, label='Training Accuracy')
ax_lstm[1].plot(epochs_lstm, test_acc_lstm, label='Testing Accuracy')
ax_lstm[1].set_title('Training & Testing Accuracy (LSTM)')
ax_lstm[1].legend()
ax_lstm[1].set_xlabel("Epochs")
plt.show()

# predicting on test data.
pred_test_lstm = model_lstm.predict(x_test)
y_pred_lstm = encoder.inverse_transform(pred_test_lstm)

y_test_original = encoder.inverse_transform(y_test)

df_lstm = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
df_lstm['Predicted Labels'] = y_pred_lstm.flatten()
df_lstm['Actual Labels'] = y_test_original.flatten()

# Save the DataFrame to a CSV file
df_lstm.to_csv('predictions_lstm.csv', index=False)
print("\nDataFrame of predictions (LSTM) saved to 'predictions_lstm.csv'")

print(df_lstm.head(10))

cm_lstm = confusion_matrix(y_test_original, y_pred_lstm)
plt.figure(figsize=(12, 10))
cm_lstm_df = pd.DataFrame(cm_lstm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
sns.heatmap(cm_lstm_df, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix (LSTM)', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()

print(classification_report(y_test_original, y_pred_lstm))

# Save the LSTM model
model_lstm.save('speech_emotion_recognition_model_lstm.h5')
print("LSTM Model saved as 'speech_emotion_recognition_model_lstm.h5'")







