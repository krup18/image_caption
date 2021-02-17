# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:59:47 2019

@author: krups
"""
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras import layers,models
import numpy as np
import time

def split_test_train_function(texts_data,test_n,val_n):
    return(texts_data[:test_n], 
           texts_data[test_n:test_n+val_n],  
           texts_data[test_n+val_n:])

def preprocessing(texts_data,images_data,max_length,vocabulary_size):
    num = len(texts_data)
    assert(num==len(images_data))
    X_text, X_image, y_text = [],[],[]
    for text,image in zip(texts_data,images_data):
        for i in range(1,len(text)):
            in_text, out_text = text[:i], text[i]
            in_text = pad_sequences([in_text],maxlen=max_length).flatten()
            out_text = to_categorical(out_text,num_classes = vocabulary_size)

            X_text.append(in_text)
            X_image.append(image)
            y_text.append(out_text)

    X_text  = np.array(X_text)
    X_image = np.array(X_image)
    y_text  = np.array(y_text)
    return(X_text,X_image,y_text)




def create_model(X_train_image,max_length,vocabulary_size):
    dim_embedding = 64
    
    input_image = layers.Input(shape=(X_train_image.shape[1],))
    fimage = layers.Dense(256,activation='relu',name="ImageFeature")(input_image)
    ## sequence model
    input_text = layers.Input(shape=(max_length,))
    ftxt = layers.Embedding(vocabulary_size,dim_embedding, mask_zero=True)(input_text)
    ftxt = layers.LSTM(256,name="CaptionFeature")(ftxt)
    ## combined model for decoder
    decoder = layers.add([ftxt,fimage])
    decoder = layers.Dense(256,activation='relu')(decoder)
    output = layers.Dense(vocabulary_size,activation='softmax')(decoder)
    model_ = models.Model(inputs=[input_image, input_text],outputs=output)
    
    model_.compile(loss='categorical_crossentropy', optimizer='adam')
    return model_


def fit_model(model_,X_train_text, X_train_image, y_train_text, X_val_text, X_val_image, y_val_text):
    start = time.time()
    hist = model_.fit([X_train_image, X_train_text], y_train_text, 
                      epochs=5, verbose=2, 
                      batch_size=64,
                      validation_data=([X_val_image, X_val_text], y_val_text))
    end = time.time()
    print("TIME TOOK {:3.2f}MIN".format((end - start )/60))
    return hist


