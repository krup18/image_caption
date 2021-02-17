# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:45:13 2019

@author: krups
"""
import random
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow import set_random_seed
import string


def set_randomseed_value(initial=123):
    np.random.seed(initial) 
    random.seed(initial)
    set_random_seed(initial)
    
def load_doc(filename):
	file = open(filename, 'r')
	text = file.read()
	file.close()
	return text

def load_descriptions(doc_text):
    textmap = []
    for l in doc_text.split('\n'):
        c = l.split('\t')
        if len(c) < 2:
            continue
        w = c[0].split("#")
        textmap.append(w + [c[1].lower()])
    return textmap
    
def word_data_frame(text_dataframe):
    vocabulary = []
    for t in text_dataframe.Caption.values:
        vocabulary.extend(t.split())
    count = Counter(vocabulary)
    a1 = []
    a2 = []
    for i in count.keys():
        a1.append(i)
    for j in count.values():
        a2.append(j)
    data = {"word":a1 , "count":a2}
    word_dataframe = pd.DataFrame(data)
    word_dataframe = word_dataframe.sort_values(by='count', ascending=False)
    word_dataframe = word_dataframe.reset_index()[["word","count"]]
    return(word_dataframe)

def clean_captions(original_caption):
    translated = str.maketrans('', '', string.punctuation)
    caption_wo_punctuation = original_caption.translate(translated)

    words_not_single_character = ""
    for word in caption_wo_punctuation.split():
        if len(word) > 1:
            words_not_single_character += " " + word

    words_not_numeric = ""
    for word in words_not_single_character.split():
        alpha = word.isalpha()
        if alpha:
            words_not_numeric += " " + word

    return(words_not_numeric)

def add_tokens(captions):
    new_captions = []
    for t in captions:
        t = 'start_seq ' + t + ' end_seq'
        new_captions.append(t)
    return(new_captions)
 
    
    
'''
jpg_dir = "Flickr8k_Dataset/Flicker8k_Dataset/"
text_dir = "Flickr8k_text/Flickr8k.token.txt"
doc_text=load_doc(text_dir)
textmap = []
load_descriptions(doc_text)

text_dataframe = pd.DataFrame(textmap,columns=["Name", "Index", "Caption"])
word_dataframe = word_data_frame(text_dataframe)

for i, caption in enumerate(text_dataframe.Caption.values):
    cleaned_caption = clean_captions(caption)
    text_dataframe["Caption"].iloc[i] = cleaned_caption

word_dataframe = word_data_frame(text_dataframe)

text1_dataframe = copy(text_dataframe)
text1_dataframe["Caption"] = add_tokens(text_dataframe["Caption"])

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
    images[name] = y_prediction.flatten()
    

images_data, index = [],[]
text1_dataframe = text1_dataframe.loc[text1_dataframe["Index"].values == "0",: ]
for i, name in enumerate(text1_dataframe.filename):
    if name in images.keys():
        images_data.append(images[name])
        index.append(i)
        
filenames = text1_dataframe["Name"].iloc[index].values
captions_data = text1_dataframe["Caption"].iloc[index].values
images_data = np.array(images_data)

nb_words = 8000
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(captions_data)
vocabulary_size = len(tokenizer.word_index) + 1

texts_data = tokenizer.texts_to_sequences(captions_data)
'''