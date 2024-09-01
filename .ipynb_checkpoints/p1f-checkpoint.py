#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import pandas as pd
import pyaudio
import wave
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect
from deep_translator import GoogleTranslator
import numpy as np
import cv2
import pytesseract
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')

# List of supported languages and their codes
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'ja': 'Japanese',
    'ko': 'Korean',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'te': 'Telugu'  # Added Telugu
}

# Load all the datasets into a list of DataFrames
datasets = [
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Lifestyle-Related Diseases.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/disease_data.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Environmental Diseases.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Idiopathic.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Neoplastic Diseases.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/non-infectious diseases_data.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Nutritional Diseases.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Psychiatric and Neurological Disorders.csv"),
    pd.read_csv("D:/AI-Projects/FinalProject/P1/Rare Diseases.csv")
]

# Combine all datasets into one DataFrame for easier searching
combined_df = pd.concat(datasets, ignore_index=True)

# Clean and normalize text
def clean_text(text):
    text = re.sub(r'\W+', ' ', text.lower())  # Remove non-alphanumeric characters and convert to lowercase
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Clean the Symptoms column in the combined DataFrame
combined_df['Symptoms'] = combined_df['Symptoms'].apply(clean_text)

# Translate text to English if not already in English
def translate_to_english(text):
    try:
        user_lang = detect(text)
        if user_lang != 'en':
            text = GoogleTranslator(source=user_lang, target='en').translate(text)
    except Exception as e:
        print(f"Error translating to English: {e}")
    return text

# Print disease information in a formatted way
def print_disease_info(disease, language='en'):
    fields = {
        'Disease Name': disease['Disease Name'],
        'Severity Level': disease['Severity Level'],
        'Symptoms': disease['Symptoms'],
        'Recommended Medications': disease['Recommended Medications'],
        'Required Food': disease['Required Food'],
        'Safety Precautions': disease['Safety Precautions'],
        'Recommended Doctor': disease['Recommended Doctor'],
        'Treatment Plan': disease['Treatment Plan'],
        'Follow-Up Recommendations': disease['Follow-Up Recommendations'],
        'Patient Education': disease['Patient Education'],
        'Recovery Time': disease['Recovery Time']
    }
    
    if language in SUPPORTED_LANGUAGES:
        try:
            fields = {key: GoogleTranslator(source='auto', target=language).translate(value) for key, value in fields.items()}
        except Exception as e:
            print(f"Error translating to {language}: {e}")
            fields = {key: value for key, value in fields.items()}
    else:
        print(f"Language code '{language}' is not supported. Showing information in English.")

    for key, value in fields.items():
        print(f"{key:<25} {value}")
    print("\n" + "="*100 + "\n")

# Identify disease based on symptoms
def identify_disease(user_input, language='en'):
    # Translate user input to English if needed
    user_input = translate_to_english(user_input)

    # Clean and extract symptoms from the user input
    cleaned_input = clean_text(user_input)

    # Combine the cleaned user input with the dataset symptoms for vectorization
    all_symptoms = combined_df['Symptoms'].tolist() + [cleaned_input]

    # Fit the TfidfVectorizer on the combined data
    tfidf = TfidfVectorizer().fit_transform(all_symptoms)

    # Separate the TF-IDF matrix for the dataset symptoms and the user's input
    dataset_tfidf = tfidf[:-1]  # All rows except the last one
    input_tfidf = tfidf[-1]  # The last row is the user's input

    # Compute cosine similarity between the user's input and the dataset symptoms
    cosine_similarities = cosine_similarity(input_tfidf, dataset_tfidf).flatten()

    # Get the index of the highest similarity
    best_match_index = cosine_similarities.argmax()

    # Check if the highest similarity is below a certain threshold
    if cosine_similarities[best_match_index] < 0.1:
        print("No close match found in the dataset. Please check your input or try different symptoms.")
    else:
        # Get the matching disease information
        matching_disease = combined_df.iloc[best_match_index]
        print_disease_info(matching_disease, language)

# Convert MP3 to WAV format
def convert_mp3_to_wav(mp3_file_path):
    wav_file_path = mp3_file_path.replace('.mp3', '.wav')
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format='wav')
        print(f"Converted MP3 file to WAV format: {wav_file_path}")
        return wav_file_path
    except Exception as e:
        print(f"Error converting MP3 to WAV: {e}")
        return None

# Record audio and save it to a file
def record_audio(output_file_path):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    
    print("Recording...")
    frames = []
    # Record for a fixed duration of 5 seconds
    for _ in range(int(44100 / 1024 * 5)):
        data = stream.read(1024)
        frames.append(data)
    
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    try:
        with wave.open(output_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(frames))
        print(f"Recorded audio saved to {output_file_path}")

        # Recognize the audio and return the recognized text
        recognizer = sr.Recognizer()
        with sr.AudioFile(output_file_path) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language='te-IN')  # Specify Telugu language code
                print(f"Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
                return ""
            except sr.RequestError as e:
                print(f"Sorry, there was an error with the speech recognition service: {e}")
                return ""

    except Exception as e:
        print(f"Error saving or processing audio file: {e}")
        return ""

# Get user input from a recorded audio file
def get_recorded_audio_input(file_path):
    file_path = file_path.strip('"')
    
    if file_path.lower().endswith('.mp3'):
        wav_file_path = convert_mp3_to_wav(file_path)
        if wav_file_path is None:
            return ""
        file_path = wav_file_path

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            print(f"Processing recorded audio file: {file_path}")
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio, language='te-IN')  # Specify Telugu language code
                print(f"Recognized text: {text}")
                return translate_to_english(text)
            except sr.UnknownValueError:
                print("Sorry, could not understand the audio.")
                return ""
            except sr.RequestError as e:
                print(f"Sorry, there was an error with the speech recognition service: {e}")
                return ""
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return ""

def identify_disease_from_image(image_file_path):
    try:
        # Load the model
        model = load_model('D:/AI-Projects/chest_xray/pneumonia_detection_model.h5')
        
        # Load and preprocess the image
        img = keras_image.load_img(image_file_path, target_size=(150, 150))  # Ensure target size matches the model input
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)  # Adjust this if needed
        
        # Predict
        preds = model.predict(x)
        
        # Interpret the prediction
        # Assuming output is a probability; adjust threshold as needed
        predicted_class = 'Pneumonia' if preds[0][0] > 0.5 else 'Normal'
        
        return predicted_class

    except Exception as e:
        print(f"Error identifying disease from image: {e}")
        return ""

# Process image input and find disease information
def process_image_input(image_file_path):
    disease_label = identify_disease_from_image(image_file_path)
    
    if disease_label:
        # Search the disease label in the combined dataset
        matched_disease = combined_df[combined_df['Disease Name'].str.contains(disease_label, case=False, na=False)]
        
        if not matched_disease.empty:
            for _, disease in matched_disease.iterrows():
                print_disease_info(disease)
        else:
            print(f"It Means Your X_Ray Was at Normal Condition: {disease_label}")

