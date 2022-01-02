# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:17:35 2020

@author: youness hnida
"""

import flask
from flask import request

import os
import librosa
import IPython.display as ipd
import numpy as np
from tensorflow.keras.models import load_model
import keras.backend.tensorflow_backend as tb
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from matplotlib import pyplot 
from sklearn.preprocessing import LabelEncoder
import subprocess
import winsound

from playsound import playsound



from zipfile import ZipFile



def predict(audio,model):
    classes=['1','2','3','4','5']
    labels=os.listdir("input/"+model);
    model=load_model('models/'+model)
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    
    return classes[index]+","+labels[index]


app = flask.Flask(__name__)
app.config["DEBUG"] = True





#cette methode pour récupérer les liste du model disponible au serveur
@app.route('/<string:name>', methods=['GET','POST'])
def method(name):
    str1=""
    str2=""
    labels=os.listdir(name)
    for label in labels:
        label=label+","
        str1=str1+label     
    str2 = str1[0:len(str1)-1]
    return str2
#cette methode pour trainer le model
@app.route('/trainning', methods=['GET','POST'])
def trainngModel():
    f = request.files['file']
    zf = ZipFile(f, 'r')
    zf.extractall('input/')
    zf.close()        
    chemin="input/"+f.filename[0:f.filename.rindex('.')]
    os.remove(chemin+"/"+f.filename)
    labels=os.listdir(chemin)
    for label in labels:
        labels_1=os.listdir(chemin+"/"+label)
        for label_1 in labels_1:  
           subprocess.call(['ffmpeg','-y', '-i',chemin+"/"+label+"/"+ label_1,chemin+"/"+label+"/"+label_1[0:label_1.rindex('.')]+'.wav'])
           os.remove(chemin+"/"+label+"/"+label_1)

    tb._SYMBOLIC_SCOPE.value = True
    f = request.files['file']
    train_audio_path = 'input/'+f.filename[0:f.filename.rindex('.')]
    labels=os.listdir(train_audio_path)
    all_wave = []
    all_label = []
    for label in labels:
        print(label)
        waves = [f for f in os.listdir(train_audio_path + '/'+ label) if f.endswith('.wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(train_audio_path + '/' + label + '/' + wav, sr = 16000)
            samples = librosa.resample(samples, sample_rate, 8000) 
            all_wave.append(samples[:8000])
            all_label.append(label) 
    le = LabelEncoder()
    y=le.fit_transform(all_label)
    classes= list(le.classes_)
    from keras.utils import np_utils
    y=np_utils.to_categorical(y, num_classes=len(labels))
    all_wave = np.array(all_wave).reshape(-1,8000,1)
    from sklearn.model_selection import train_test_split
    x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_wave),np.array(y),stratify=y,test_size = 0.2,random_state=777,shuffle=True)
    
    K.clear_session()
    inputs = Input(shape=(8000,1))
    
    #First Conv1D layer
    conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Second Conv1D layer
    conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Third Conv1D layer
    conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Fourth Conv1D layer
    conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
    conv = MaxPooling1D(3)(conv)
    conv = Dropout(0.3)(conv)
    
    #Flatten layer
    conv = Flatten()(conv)
    
    #Dense Layer 1
    conv = Dense(256, activation='relu')(conv)
    conv = Dropout(0.3)(conv)
    
    #Dense Layer 2
    conv = Dense(128, activation='relu')(conv)
    conv = Dropout(0.3)(conv)
    
    outputs = Dense(len(labels), activation='softmax')(conv)
    
    model = Model(inputs, outputs)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001) 
    mc = ModelCheckpoint(f.filename[0:f.filename.rindex('.')]+'.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    history=model.fit(x_tr, y_tr ,epochs=100, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))
    model.save("models/"+f.filename[0:f.filename.rindex('.')])
    pyplot.plot(history.history['loss'], label='train') 
    pyplot.plot(history.history['val_loss'], label='test') 
    pyplot.legend() 
    pyplot.show()
    
    return 'this model is trained :'+f.filename[0:f.filename.rindex('.')]


#cette methode pour test le model
@app.route('/', methods=['GET','POST'])
def home():

       f = request.files['file']
       f.save(f.filename)
       
       subprocess.call(['ffmpeg','-y', '-i', f.filename,'file.wav']) 
       samples, sample_rate = librosa.load('file.wav', sr = 16000)
       samples = librosa.resample(samples, sample_rate, 8000)
       ipd.Audio(samples,rate=8000) 
       #f.filename
       #f.save(f.filename)

       filename = 'file.wav'
       winsound.PlaySound(filename, winsound.SND_FILENAME)                     
       return predict(samples[:8000],request.form.get('model'))


app.run(host='192.168.1.6',debug=False, threaded=True) 