import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
import keras
import warnings
warnings.filterwarnings("ignore")

def CalculateMelSpectrogram(file_location):
    y, sr = librosa.load(file_location)
    melSpec = librosa.feature.melspectrogram(y=y, sr=sr)
    melSpec_dB = librosa.power_to_db(melSpec)
    dim = (32, 32)
    resized = cv2.resize(melSpec_dB, dim, interpolation = cv2.INTER_AREA)
    return resized

def init_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def urdu_digit_classifier():
    file_path = 'assets/inference_audios/file4.wav'
    model_path = 'models/urdu_model.h5'
    img = CalculateMelSpectrogram(file_location=file_path)
    x = np.array(img)
    x = x.reshape((1,32,32,1))
    model = init_model(model_path)
    print(model.predict(x))