import random
import numpy as np
import nltk
import sklearn
import keras
import json
import scipy
from collections import defaultdict
import gen_util
import keras
random.seed(1337)

path="../big_domain_desc/"
maxlen=120
offset1=100    #we increase the training examples.
set_mode='train1'
num_train=900
num_valid=100
top_k=20

with open("../meta_data/clean_big_domain_desc_split.json") as f:
    split=json.load(f)
    classes=split[set_mode]
    
train_corpus={}
corpus=[]
for cls in classes:
    with open(path+cls+".txt") as f:
        samples=f.readlines()
        random.shuffle(samples)
    train_examples=samples[:offset1]
    corpus.extend(train_examples )
    train_corpus[cls]=[nltk.tokenize.word_tokenize(doc) for doc in train_examples ]
    
with open("word_idx.json") as f:
    word_idx=json.load(f)
    
model=keras.models.load_model("../../model/encoder_lstm_512.h5")

train_rep, train_cls_rep=[], []
for cls in classes:
    train_idx=[[word_idx[token] if token in word_idx else 1 for token in doc] for doc in train_corpus[cls] ]
    train_idx=keras.preprocessing.sequence.pad_sequences(train_idx, maxlen=maxlen, padding="post")
    cls_rep=model.predict(train_idx)
    train_rep.append(cls_rep)
    train_cls_rep.append(cls_rep.mean(axis=0) )
train_rep=np.vstack(train_rep)
train_cls_rep=np.vstack(train_cls_rep )

class_set=classes
train_X0, train_X1, train_Y=gen_util.data2np_train_idx_neg_cls(class_set[:num_train], train_rep, train_cls_rep, classes, offset1, top_k)
valid_X0, valid_X1, valid_Y=gen_util.data2np_train_idx_neg_cls(class_set[-num_valid:], train_rep, train_cls_rep, classes, offset1, top_k)

np.savez("../data/"+set_mode+"_idx.npz", 
         train_rep=train_rep, #including all validation examples.
         train_X0=train_X0, train_X1=train_X1, train_Y=train_Y, 
         valid_X0=valid_X0, valid_X1=valid_X1, valid_Y=valid_Y)