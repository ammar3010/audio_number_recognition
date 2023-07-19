import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import librosa
import matplotlib.pyplot as plt
import os
from record_audio import record

def route_to_numpy_array(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (196, 256))
    resized_image = resized_image.astype('float32') / 255
    return resized_image

def initialize_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def execute(model, filename, img_path):
    audio_name = filename
    df = pd.DataFrame(columns=["filename","image"])
    df.at[0,"filename"] = filename
    print("here")
    plt.figure(figsize=(3,5))
    path = img_path + '/' + df["filename"][0]
    audio, sample_rate = librosa.load(path, sr=None, mono=False)
    sgram = librosa.stft(audio, n_fft=1024, hop_length=None)
    simple_sgram, _ = librosa.magphase(sgram)
    sgram_mel = librosa.feature.melspectrogram(S=simple_sgram, sr=sample_rate, n_mels=64)
    sgram_amp_log = librosa.amplitude_to_db(sgram_mel, ref=np.min)
    filename = 'assets/.inference_audios/' + df['filename'][0].split('.')[0] + '.jpg'
    librosa.display.specshow(sgram_amp_log, sr=sample_rate/2, x_axis="time",y_axis="mel")
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

    df.at[0,"image"] = route_to_numpy_array(filename)
    h = df["image"].values
    h: np.ndarray = np.stack(h)
    h = h.reshape((len(h), 196, 256, 1))
    transformed = h
    prediction = model.predict(transformed)

    preds = prediction[0].tolist()
    max = 0
    index = 0
    print(prediction)
    for i in range(len(preds)):
        if preds[i] > max:
            max = preds[i]
            index = i
            
    print(index)
    os.remove('assets/.inference_audios/'+audio_name)
    os.remove(filename)
    
if __name__ == "__main__":
    model_path = 'models/my_model.h5'
    filename = record()
    img_path = 'assets/.inference_audios/'
    model = initialize_model(model_path)
    execute(model,filename,img_path)