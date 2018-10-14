# Author: Avinash Madasu

# Module: MaLSTM Model

# Competition : Quora question pairs

#packages required
import os
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import gensim
import nltk
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

#Download these if nltk doesn't have
nltk.download('punkt')
nltk.download('stopwords')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stopword = stopwords.words('english')

#Train data
train_data = pd.read_csv('train.csv')

print(len(train_data))

train_data.drop(["qid1","qid2","id"],inplace=True,axis=1)

train_labels = train_data["is_duplicate"].astype(int)

train_labels = train_labels.as_matrix()

print(train_labels[0:2])


# Load Google pre trained vectors:
# Mention the correct path to your bin/txt file 
google_w2v_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)


# This is the stanford Glove Model download 'glove.42B.300d.zip',Zip it and keep in your directory. Each vector is 300 dimension
def load_glove_model():
  glove_model = {}
  with open('glove.42B.300d.txt','r') as f: # Path to your Glove File
    for line in f.readlines():
      splitline = line.split()
      word = splitline[0]
      embedding = np.array([float(val) for val in splitline[1:]])
      glove_model[word] = embedding
   
  return glove_model

#This function loads the glove model
glove_w2v_model = load_glove_model() 

# Preprocess the data using this function.
# It return list of tokens after preprocessing 
def preprocess(text): 
  text = re.sub(r"it\'s","it is",str(text))
  text = re.sub(r"i\'d","i would",str(text))
  text = re.sub(r"don\'t","do not",str(text))
  text = re.sub(r"he\'s","he is",str(text))
  text = re.sub(r"there\'s","there is",str(text))
  text = re.sub(r"that\'s","that is",str(text))
  text = re.sub(r"can\'t", "can not", text)
  text = re.sub(r"cannot", "can not ", text)
  text = re.sub(r"what\'s", "what is", text)
  text = re.sub(r"What\'s", "what is", text)
  text = re.sub(r"\'ve ", " have ", text)
  text = re.sub(r"n\'t", " not ", text)
  text = re.sub(r"i\'m", "i am ", text)
  text = re.sub(r"I\'m", "i am ", text)
  text = re.sub(r"\'re", " are ", text)
  text = re.sub(r"\'d", " would ", text)
  text = re.sub(r"\'ll", " will ", text)
  text = re.sub(r"\'s"," is",text)
  text = re.sub(r"[0-9]"," ",str(text))
  sents = word_tokenize(text)
  return sents

# Divide the training data into Train set and Validation set

# Train data --> 98%
# Validation data --> 2%
X_train,X_val,y_train,y_val = train_test_split(train_data,train_labels,test_size=0.02,random_state=0)


# Since the split is done on pandas dataframe the indices need to be reset to begin from 0
# This function resets the indices

def resetindex(data):
  data.reset_index(drop=True,inplace=True)
  return data

X_train = resetindex(X_train)  # Reset the train set indices
X_val = resetindex(X_val)     # Reset the validation set indices

train_length = len(X_train)
val_length = len(X_val)

max_sentence_length = 20  # Maximum number of words per sentence to be considered
embedding_dim = 300     # Each word is converted to 300 dimensional vector

train_ques1 = np.zeros((train_length,max_sentence_length,embedding_dim))  # Vectors of question1 in train set
train_ques2 = np.zeros((train_length,max_sentence_length,embedding_dim))  # Vectors of question2 in train set

val_ques1 = np.zeros((val_length,max_sentence_length,embedding_dim))  # Vectors of question1 in validation set
val_ques2 = np.zeros((val_length,max_sentence_length,embedding_dim))  # Vectors of question2 in validation set

# This function is to add padding to sentences if the sentence length is less than max_sentence_length
# There are 2 types of intializations for words not in both the word vector models
# 1. Intialize with some random numbers
# 2. Intialize with zeros
# I have intizlized to zeros because it gave better results than random intialization.
# You can uncomment if you want to intialize randomly
# You can also intialize using normal distribution -----> np.random.normal(0,1,embedding_dim)


#def pad_sentences(start,vectors):
 # for i in range(start,max_sentence_length):
 #   vectors[i,:] = np.random.rand(embedding_dim)

# This function checks if word is in both pretrained models
# If word exits then it is added and count is increased
# If word doesn't exists it intialized to zero if the count is less than max_sentence_length

# You can also use Embedding layer of keras
# For that you have to create a dictionary of words that doesn't exist in Word2vec models
# Then you have to create indices for Out of Vocabulary words 
# Also the question number has to bes stored along with words to map corresponding vectors to that question

# This method is easier.

def convert_to_vectors(sentence):
  sents = preprocess(sentence)
  vectors = np.zeros((max_sentence_length,embedding_dim))
  count = 0
  for sent in sents:
    if sent not in stopword:   # Check if word is not a stopword
      if sent in glove_w2vmodel:    # Check if word is in glove model
        vectors[count,:] = glove_w2vmodel[sent]
        count+=1
      elif sent in google_w2v_model:  # Check if word is in google word2vec pretrained model
        vectors[count,:] = google_w2v_model[sent]
        count+=1
    if(count==max_sentence_length):   # If count of words equals max_sentence_length return vectors
      return vectors
  #if(count<max_sentence_length):  # Uncomment this pad_sentences function is uncommented
   # pad_sentences(count,vectors)
    #return vectors

def generate_train_vectors():
  for i in range(train_length):
    train_ques1[i,:,:] = convert_to_vectors(X_train["question1"][i])
    train_ques2[i,:,:] = convert_to_vectors(X_train["question2"][i])

# Generate vectors for Train set

generate_train_vectors()  

def generate_validation_vectors():
  for i in range(val_length):
    val_ques1[i,:,:] = convert_to_vectors(X_val["question1"][i])
    val_ques2[i,:,:] = convert_to_vectors(X_val["question2"][i])

# Generate vectors for validation set

generate_validation_vectors()

# This is the similarity measure used to compare two sentences

def ManhattanDistance(l1,l2):
  return K.exp(-K.sum(K.abs(l1-l2), axis=1, keepdims=True))

# Defining LSTM model

def generate_model(n_hidden=64):
  left_input = Input(shape=(max_sentence_length,embedding_dim))
  right_input = Input(shape=(max_sentence_length,embedding_dim))
  

  lstm = LSTM(n_hidden,return_sequences=False)

  left_output = lstm(left_input)
  right_output = lstm(right_input)

  manhattan_distance = Lambda(lambda x:ManhattanDistance(x[0],x[1]),output_shape=lambda x:(x[0][0],1))([left_output,right_output])

  model = Model(inputs=[left_input,right_input],outputs=[manhattan_distance])

  # Adadelta is used as optimizer and clipnorm is used to prevent gradient explosion
  # Clipnorm normalizes the gradients exceeding certain threshold

  optimizer = Adadelta(clipnorm=1.25)
  
  # Mean squared loss is used to measure loss
  model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['accuracy'])

  return model


# hidden size of LSTM
n_hidden = 64
model = generate_model(n_hidden)

model.fit([train_ques1,train_ques2],y_train,validation_data=([val_ques1,val_ques2],y_val),batch_size=64,epochs=30)
model.save('manhattan-lstm.h5')


test_data = pd.read_csv('test.csv')

test_length = len(test_data)

print(test_length)

print(test_data.head(3))

test_data.drop(["test_id",],axis=1,inplace=True)

test_ques1 = np.zeros((test_length,max_sentence_length,embedding_dim))
test_ques2 = np.zeros((test_length,max_sentence_length,embedding_dim))

def generate_test_vectors():
  for i in range(test_length):
    test_ques1[i,:,:] = convert_to_vectors(test_data["question1"][i])
    test_ques2[i,:,:] = convert_to_vectors(test_data["question2"][i])

generate_test_vectors()

pred = model.predict([test_ques1,test_ques2],batch_size=4096)

predictions = np.zeros(test_length,dtype='int32')
for i in range(test_length):
  if(pred[i]>=0.5):
    predictions[i] = int(1)

print(len(predictions))

test = pd.DataFrame({'is_duplicate':predictions})

print(len(test))

test.to_csv('predictions.csv',header=True,index_label='test_id')


