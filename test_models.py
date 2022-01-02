# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:17:35 2020

@author: youness hnida
"""

import os
import librosa
import numpy as np
import IPython.display as ipd
import numpy as np
from keras.models import load_model

#train_audio_path = 'Models/Model Arabic Maroc (Darija)'
#classes=os.listdir(train_audio_path)
classes=["asir","liasar","liman","ou9af","rjae"]
model=load_model('Models/Model Arabic Maroc (Darija)')

def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]

#reading the voice commands
samples, sample_rate = librosa.load('audio.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate=8000)  
print(predict(samples[:8000]))