import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import keras
from keras.layers import Dense,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input
from keras.models import Sequential,Model
from keras import metrics
from keras.layers import LeakyReLU,BatchNormalization
from keras.layers.merge import concatenate,Concatenate
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.models import load_model
from keras.layers import Lambda


nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stopword  = stopwords.words('english')

data = pd.read_csv('train.csv')

train_labels = data['is_duplicate'].astype(int)
train_labels = train_labels.as_matrix()

def load_data(file):
  data = pd.read_csv(file)
  data = data[['question1','question2']]
  ques1 = list(data['question1'].as_matrix())
  ques2 = list(data['question2'].as_matrix())
  del data
  return ques1,ques2

t_ques1,t_ques2,train_labels = load_data('train.csv')

train_length = len(t_ques1)

assert (len(t_ques1)==len(t_ques2))
'''
def load_glove_model():
  model = {}
  with open('glove.42B.300d.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
      sentence = line.split()
      word,embedding = sentence[0],np.array([float(value) for value in sentence[1:]])
      model[word] = embedding
    return model

glove_w2v = load_glove_model()
'''
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
  text = re.sub(r"[^a-zA-Z]"," ",str(text))
  sents = word_tokenize(text)
  new_sentence = " "
  for word in sents:
    if word.lower() not in stopword:
      new_sentence+=word.lower()+" "
  return new_sentence

train_q1 = []
train_q2 = []

for i in range(train_length):
  train_q1.append(preprocess(t_ques1[i]))
  train_q2.append(preprocess(t_ques2[i]))

train_q = train_q1 + train_q2

print(train_q[0:2])

max_sentence_length = 25
embedding_dim = 300
max_num_words = 200000

def make_features():
  tokenizer =  Tokenizer(num_words=max_num_words)
  tokenizer.fit_on_texts(train_q)
  ques1_tokens = tokenizer.texts_to_sequences(train_q1)
  ques2_tokens = tokenizer.texts_to_sequences(train_q2)
  train_q1_fea = pad_sequences(ques1_tokens,maxlen=max_sentence_length)
  train_q2_fea = pad_sequences(ques2_tokens,maxlen=max_sentence_length)
  return train_q1_fea,train_q2_fea

train_q1_fea,train_q2_fea = make_features()

train_q1.shape

def create_model():
  train_ques1 = Input(shape=(max_sentence_length,))
  train_ques2 = Input(shape=(max_sentence_length,))
  q1_embedding = Embedding(input_dim=max_num_words,output_dim=embedding_dim,trainable=False,input_length=max_sentence_length)(train_ques1)
  q2_embedding = Embedding(input_dim=max_num_words,output_dim=embedding_dim,trainable=False,input_length=max_sentence_length)(train_ques2)
  ques1_dense = Dense(embedding_dim,activation='relu')(q1_embedding)
  ques2_dense = Dense(embedding_dim,activation='relu')(q2_embedding)
  ques1_output = Lambda(lambda x:K.max(x,axis=1,keepdims=False),output_shape=(embedding_dim,))(ques1_dense)
  ques2_output = Lambda(lambda x:K.max(x,axis=1,keepdims=False),output_shape=(embedding_dim,))(ques2_dense)
  combined_output = Concatenate()([ques1_output,ques2_output])
  
  dense1 = Dense(200,activation='relu')(combined_output)
  dense1_drop = Dropout(0.1)(dense1)
  dense1_out = BatchNormalization()(dense1_drop)
  dense2 = Dense(200,activation='relu')(dense1_out)
  dense2_drop = Dropout(0.1)(dense2)
  dense2_out = BatchNormalization()(dense2_drop)
  dense3 = Dense(200,activation='relu')(dense2_out)
  dense3_drop = Dropout(0.1)(dense3)
  dense3_out = BatchNormalization()(dense3_drop)
  preds = Dense(1,activation='sigmoid')(dense3_out)
  
  model = Model(inputs=[train_ques1,train_ques2],outputs=preds)
  
  model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
  
  return model

siamese_dense_model = create_model()

siamese_dense_model.fit([train_q1_fea,train_q2_fea],train_labels,batch_size=32,epochs=25)

siamese_dense_model.save('siamese_dense_model.h5')
'''
def build_features(sentence1,sentence2):
  sent1_tokens = tokenizer.texts_to_sequences(sentence1)
  sent2_tokens = tokenizer.texts_to_sequences(sentence2)
  sent1_fea = pad_sequences(sent1_tokens,maxlen=max_sentence_length)
  sent2_fea = pad_sequences(sent2_tokens,maxlen=max_sentence_length)
  return sent1_fea,sent2_fea

def make_predict(sentence1,sentence2):
  #siamese_dense_model = create_model()
  #siamese_dense_model.load('siamese_dense_model.h5')
  sentence1_fea,sentence2_fea = build_features(sentence1,sentence2)
  prediction = siamese_dense_model.predict([sentence1_fea,sentence2_fea])
  print(prediction)
  #if prediction>=0.5:
  #  return 1
  #else:
  return 0
'''
test_data = pd.read_csv('test.csv')

test_length = len(test_data)

print(test_data.head(3))

test_data.drop(["test_id"],axis=1,inplace=True)

test_ques1 = test_data['question1'].as_matrix()
test_ques2 = test_data['question2'].as_matrix()

test_q1 = []
test_q2 = []

for i in range(test_length):
  test_q1.append(preprocess(test_ques1[i]))
  test_q2.append(preprocess(test_ques2[i]))

test_data.drop(['question1','question2'],axis=1,inplace=True)

test_ques1_tokens = tokenizer.texts_to_sequences(test_q1)
test_ques2_tokens = tokenizer.texts_to_sequences(test_q2)
test_q1_fea = pad_sequences(test_ques1_tokens,maxlen=max_sentence_length)
test_q2_fea = pad_sequences(test_ques2_tokens,maxlen=max_sentence_length)

test_labels = np.zeros(test_length)

predictions = siamese_dense_model.predict([test_q1_fea,test_q2_fea],batch_size=4096)

for i in range(test_length):
  if(predictions[i]>=0.5):
    test_labels[i] = 1

test = pd.DataFrame({'is_duplicate':test_labels})

test.to_csv('siamese_dense.csv',header=True,index_label='test_id')


