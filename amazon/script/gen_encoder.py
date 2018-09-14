import os
import random
import numpy as np
import nltk
import sklearn
import keras
import json
import scipy
from collections import defaultdict
import gen_util
random.seed(1337)

maxlen=120

path="../big_domain_desc_encoder/"
with open("../meta_data/clean_big_domain_desc_split.json") as f:
    split=json.load(f)
    classes=split['train']
    
train_examples={}
train_corpus={}
corpus=[]
for cls in classes:
    with open(path+cls+".txt") as f:
        samples=f.readlines()
        random.shuffle(samples)
    train_examples[cls]=samples
    corpus.extend(train_examples[cls] )
    #tokenize the corpus
    train_corpus[cls]=[nltk.tokenize.word_tokenize(doc) for doc in train_examples[cls] ]
    
with open("word_idx.json") as f:
    word_idx=json.load(f)
    
train_idx={}
for cls in classes:
    train_idx[cls]=[[word_idx[token] if token in word_idx else 1 for token in doc] for doc in train_corpus[cls]]
    train_idx[cls]=keras.preprocessing.sequence.pad_sequences(train_idx[cls], maxlen=maxlen, padding="post")
    
def data2np_s_train(class_set, train_idx):
    train_X, train_Y=[], []
    for ix, cls in enumerate(class_set):
        train_X.append(train_idx[cls] )
        train_Y.append(np.full(len(train_idx[cls]), ix) )
    train_X=np.vstack(train_X)
    train_Y=np.concatenate(train_Y)
    shuffle_idx=np.random.permutation(train_X.shape[0])
    return train_X[shuffle_idx], train_Y[shuffle_idx]

class_set=classes
s_train_X, s_train_Y=data2np_s_train(class_set, train_idx)
np.savez("../data/encoder.npz", s_train_X=s_train_X, s_train_Y=s_train_Y)