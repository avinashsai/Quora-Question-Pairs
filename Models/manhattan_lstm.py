# Author: Avinash Madasu
import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

import zipfile

import nltk
nltk.download('punkt')
nltk.download('stopwords')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stopword = stopwords.words('english')

with zipfile.ZipFile('train.csv.zip', 'r') as myzip:
    myzip.extractall()

with zipfile.ZipFile('test.csv.zip', 'r') as myzip:
    myzip.extractall()

train_data = pd.read_csv('train.csv')

print(len(train_data))


with zipfile.ZipFile('glove.42B.300d.zip', 'r') as myzip:
    myzip.extractall()

def load_glove_model():
  glove_model = {}
  with open('glove.42B.300d.txt','r') as f:
    for line in f.readlines():
      splitline = line.split()
      word = splitline[0]
      embedding = np.array([float(val) for val in splitline[1:]])
      glove_model[word] = embedding
   
  return glove_model

glove_model = load_glove_model()

print(train_data.head())

train_data.drop(["qid1","qid2","id"],inplace=True,axis=1)

train_labels = train_data["is_duplicate"].astype(int)

train_labels = train_labels.as_matrix()

print(train_labels[0:2])

train_length = len(train_data)

def preprocess(sentence):
  sentence = re.sub(r"can\'t","can not",str(sentence))
  sentence = re.sub(r"n\'t"," not",str(sentence))
  sentence = re.sub(r"\'ll"," will",str(sentence))
  sentence = re.sub(r"\'s"," is",str(sentence))
  sentence = re.sub(r"\'am"," am",str(sentence))
  sentence = re.sub(r"\'ve"," have",str(sentence))
  sentence = re.sub(r"\'d"," would",str(sentence))
  sentence = re.sub(r'[^a-zA-Z]'," ",str(sentence))
  sentence = sentence.split()
  return sentence

def convert_to_vectors(sentence):
  sentence = preprocess(sentence)
  length = len(sentence)
  vectors = np.zeros(300)
  count = 0
  for sent in sentence:
    if(sent not in stopword):
      if(sent in glove_model):
        vectors+=glove_model[sent]
      else:
        vectors+=np.random.normal(0,1,300)
  
  if(length>0):
    vectors = vectors/length
    vectors = vectors[:50]
  else:
    vectors = np.random.normal(0,1,50)
  
  return vectors.reshape((50,1))

train_ques1 =  np.zeros((train_length,50,1))
train_ques2 =  np.zeros((train_length,50,1))

for i in range(train_length):
  train_ques1[i] = convert_to_vectors(train_data["question1"][i])
  train_ques2[i] = convert_to_vectors(train_data["question2"][i])

import tensorflow as tf
import keras

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Merge,Bidirectional
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import LeakyReLU,Dense,Dropout,Lambda

from keras import metrics

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import concatenate

def ManhattanDistance(l1,l2):
  return K.exp(-K.sum(K.abs(l1-l2), axis=1, keepdims=True))

left_input = Input(shape=(50,1))
right_input = Input(shape=(50,1))

n_hidden = 64

lstm = LSTM(n_hidden,return_sequences=False)

left_output = lstm(left_input)
right_output = lstm(right_input)

manhattan_distance = Lambda(lambda x:ManhattanDistance(x[0],x[1]),output_shape=lambda x:(x[0][0],1))([left_output,right_output])

model = Model(inputs=[left_input,right_input],outputs=[manhattan_distance])

optimizer = Adadelta(clipnorm=1.25)

model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

model.fit([train_ques1,train_ques2],train_labels,batch_size=1024,epochs=15)
model.save('siamese-lstm.h5')

test_data = pd.read_csv('test.csv')

test_length = len(test_data)

print(test_length)

print(test_data.head(3))

test_data.drop(["test_id",],axis=1,inplace=True)

test_ques1 = np.zeros((test_length,50,1))
test_ques2 = np.zeros((test_length,50,1))

for i in range(test_length):
  test_ques1[i] = convert_to_vectors(test_data["question1"][i])
  test_ques2[i] = convert_to_vectors(test_data["question2"][i])

pred = model.predict([test_ques1,test_ques2],batch_size=4096)

predictions = np.zeros(test_length,dtype='int32')
for i in range(test_length):
  if(pred[i]>=0.5):
    predictions[i] = int(1)

print(len(predictions))

preds = predictions[0:2345797]


test = pd.DataFrame({'is_duplicate':preds})

print(len(test))

test.to_csv('predictions.csv',header=True,index_label='test_id')


