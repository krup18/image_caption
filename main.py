# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:55:09 2019

@author: krups
"""
import preprocessing
import mymodel
import evaluation
import numpy as np
from copy import copy
from keras.applications import VGG16
from keras import models
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from collections import OrderedDict
from keras.preprocessing.text import Tokenizer
from keras.backend.tensorflow_backend import set_session
import os, warnings
import pandas as pd 
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu
warnings.filterwarnings("ignore")

settings = tf.ConfigProto()
settings.gpu_options.per_process_gpu_memory_fraction = 0.95
settings.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=settings))


jpg_dir = "Flickr8k_Dataset/Flicker8k_Dataset/"
text_dir = "Flickr8k_text/Flickr8k.token.txt"
doc_text=preprocessing.load_doc(text_dir)
textmap=preprocessing.load_descriptions(doc_text)

text_dataframe = pd.DataFrame(textmap,columns=["Name", "Index", "Caption"])
word_dataframe = preprocessing.word_data_frame(text_dataframe)

for i, caption in enumerate(text_dataframe.Caption.values):
    cleaned_caption = preprocessing.clean_captions(caption)
    text_dataframe["Caption"].iloc[i] = cleaned_caption

word_dataframe = preprocessing.word_data_frame(text_dataframe)

text1_dataframe = copy(text_dataframe)
text1_dataframe["Caption"] = preprocessing.add_tokens(text_dataframe["Caption"])

vgg = VGG16(include_top=True,weights=None)
vgg.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels.h5")
vgg.summary()
vgg.layers.pop()
vgg = models.Model(inputs=vgg.inputs, outputs=vgg.layers[-1].output)
#vgg.summary()


images = OrderedDict()
pixels = 224
t_size = (pixels,pixels,3)
pics = os.listdir(jpg_dir)
data = np.zeros((len(pics),pixels,pixels,3))
for index, name in enumerate(pics):
    file_name = jpg_dir + '/' + name
    img = load_img(file_name, target_size=t_size)
    img = img_to_array(img)
    nimg = preprocess_input(img)   
    y_prediction = vgg.predict(nimg.reshape( (1,) + nimg.shape[:3]))
    print(index)
    images[name] = y_prediction.flatten()
    

images_data, index = [],[]
text1_dataframe = text1_dataframe.loc[text1_dataframe["Index"].values == "0",: ]
for i, name in enumerate(text1_dataframe.Name):
    if name in images.keys():
        images_data.append(images[name])
        index.append(i)
        
filenames = text1_dataframe["Name"].iloc[index].values
captions_data = text1_dataframe["Caption"].iloc[index].values
images_data = np.array(images_data)

#nb_words = 8000
tokenizer = Tokenizer(nb_words=8000)
tokenizer.fit_on_texts(captions_data)
vocabulary_size = len(tokenizer.word_index) + 1

texts_data = tokenizer.texts_to_sequences(captions_data)



test_n, val_n = int(len(texts_data)*0.2), int(len(texts_data)*0.2)
test_data_text,  val_data_text, train_data_text   = mymodel.split_test_train_function(texts_data,test_n,val_n)
test_data_image,  val_data_image, train_data_image   = mymodel.split_test_train_function(images_data,test_n,val_n)
test_fnames,val_fnames, train_fnames  = mymodel.split_test_train_function(filenames,test_n,val_n)

max_length = np.max([len(text) for text in texts_data])

X_train_text, X_train_image, y_train_text = mymodel.preprocessing(train_data_text,train_data_image,max_length,vocabulary_size)
X_val_text,   X_val_image,   y_val_text   = mymodel.preprocessing(val_data_text,val_data_image,max_length,vocabulary_size)

model_=mymodel.create_model(X_train_image,max_length,vocabulary_size)
hist=mymodel.fit_model(model_,X_train_text, X_train_image, y_train_text,X_val_text,   X_val_image,   y_val_text)


index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])





nkeep = 5
pred_good, pred_bad, bleus = [], [], [] 
count = 0 
for jpgfnm, image_feature, tokenized_text in zip(test_fnames,test_data_image,test_data_text):
    count += 1
    if count % 200 == 0:
        print("  {:4.2f}% is done..".format(100*count/float(len(test_fnames))))
    
    caption_true = [ index_word[i] for i in tokenized_text ]     
    caption_true = caption_true[1:-1] ## remove startreg, and endreg
    ## captions
    caption = evaluation.predict_caption(image_feature.reshape(1,len(image_feature)),max_length,tokenizer,index_word,model_)
    caption = caption.split()
    caption = caption[1:-1]## remove startreg, and endreg
    
    bleu = sentence_bleu([caption_true],caption)
    bleus.append(bleu)
    if bleu > 0.7 and len(pred_good) < nkeep:
        pred_good.append((bleu,jpgfnm,caption_true,caption))
    elif bleu < 0.3 and len(pred_bad) < nkeep:
        pred_bad.append((bleu,jpgfnm,caption_true,caption))
        
print("Mean BLEU {:4.3f}".format(np.mean(bleus)))

print("Bad Caption")
evaluation.plot_images(pred_bad,jpg_dir)
print("Good Caption")
evaluation.plot_images(pred_good,jpg_dir)




test_image_location="test.jpg"
npix = 224
target_size = (npix,npix,3)    
count = 1
testimage = load_img(test_image_location, target_size=target_size)
testimage = img_to_array(testimage)
ntestimage = preprocess_input(testimage)


ytest_pred = vgg.predict(ntestimage.reshape( (1,) + ntestimage.shape[:3]))
testimages = ytest_pred.flatten()
dtestimages = np.array(testimages)
testcaption = evaluation.predict_caption(dtestimages.reshape(1,len(dtestimages)),max_length,tokenizer,index_word,model_)

print(testcaption)
