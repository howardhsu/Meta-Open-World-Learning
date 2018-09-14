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
train_per_cls=100 #seen examples
test_per_cls=50 #test examples so 5000 test examples in the end.
mode="test"

with open("../meta_data/clean_big_domain_desc_split.json") as f:
    split=json.load(f)
    classes=split[mode]
    
train_corpus, test_corpus={}, {} #let's not change the name of train_corpus, but it's actually seen_corpus 
corpus=[]
for cls in classes:
    with open(path+cls+".txt") as f:
        samples=f.readlines()
        random.shuffle(samples)
    train_examples=samples[:train_per_cls]
    test_examples=samples[-test_per_cls:]
    corpus.extend(train_examples )
    corpus.extend(test_examples )
    train_corpus[cls]=[nltk.tokenize.word_tokenize(doc) for doc in train_examples ]
    test_corpus[cls]=[nltk.tokenize.word_tokenize(doc) for doc in test_examples ]
    
with open("word_idx.json") as f:
    word_idx=json.load(f)
    
model=keras.models.load_model("../../model/encoder_lstm_512.h5")

train_idx, test_idx, train_Y=[], [], []
train_rep, test_rep=[], []
for ix, cls in enumerate(classes):
    tmp_idx=[[word_idx[token] if token in word_idx else 1 for token in doc] for doc in train_corpus[cls] ]
    tmp_idx=keras.preprocessing.sequence.pad_sequences(tmp_idx, maxlen=maxlen, padding="post")
    train_idx.append(tmp_idx)
    train_rep.append(model.predict(tmp_idx) )
    train_Y.append(np.full((train_per_cls,), ix) )
    tmp_idx=[[word_idx[token] if token in word_idx else 1 for token in doc] for doc in test_corpus[cls] ]
    tmp_idx=keras.preprocessing.sequence.pad_sequences(tmp_idx, maxlen=maxlen, padding="post")
    test_idx.append(tmp_idx)
    test_rep.append(model.predict(tmp_idx) )
train_idx=np.vstack(train_idx)
train_Y=np.concatenate(train_Y, 0)
train_rep=np.vstack(train_rep)
test_idx=np.vstack(test_idx)
test_rep=np.vstack(test_rep)

def data2np_DOC_train(class_set, data_idx, data_rep, data_Y, classes, train_per_cls):
    train_idx, train_rep, train_Y=[], [], []
    for cls in class_set:
        ix=classes.index(cls)
        cls_offset=ix*train_per_cls
        train_idx.append(data_idx[cls_offset:cls_offset+train_per_cls] )
        train_rep.append(data_rep[cls_offset:cls_offset+train_per_cls] )
        train_Y.append(data_Y[cls_offset:cls_offset+train_per_cls])
    train_idx=np.vstack(train_idx)
    train_rep=np.vstack(train_rep)
    train_Y=np.concatenate(train_Y, 0)
    shuffle_idx=np.random.permutation(train_idx.shape[0])
    return train_idx[shuffle_idx], train_rep[shuffle_idx], train_Y[shuffle_idx]

def data2np_test(class_set, train_rep, test_idx, test_rep, classes, train_per_cls, test_per_cls):
    test_X1, test_Y=[], []
    test_X0=test_rep #testing over the full test examples. we only change the dim 1 of test_X1 for range of known classes.
    test_idx_X0=test_idx
    for cls in class_set: #looping through train classes
        ix=classes.index(cls)
        cls_offset=ix*train_per_cls
        sim=sklearn.metrics.pairwise.cosine_similarity(test_X0, train_rep[cls_offset:cls_offset+train_per_cls ])
        sim_idx=sim.argsort(axis=1)[:,1:]+cls_offset #keep all 99 examples
        test_X1.append(np.expand_dims(sim_idx, 1) )
    test_X1=np.concatenate(test_X1, 1)
    #build the rejection class.
    test_Y=np.concatenate([np.repeat(np.arange(len(class_set) ), test_per_cls), np.full( test_per_cls*(len(classes)-len(class_set) ) , len(class_set) ) ])
    return test_idx_X0, test_X0, test_X1, test_Y

for cut in [25, 50, 75]:
    class_set=classes[:cut]
    train_set_idx_X, train_set_X, train_set_Y=data2np_DOC_train(class_set, train_idx, train_rep, train_Y, classes, train_per_cls)
    test_idx_X0, test_X0, test_X1, test_Y=data2np_test(class_set, train_rep, test_idx, test_rep, classes, train_per_cls, test_per_cls)
    np.savez("../data/"+mode+"_"+str(cut)+"_idx.npz", 
             train_rep=train_rep, 
             train_set_idx_X=train_set_idx_X, train_set_X=train_set_X, train_set_Y=train_set_Y,
             test_idx_X0=test_idx_X0, test_X0=test_X0, test_X1=test_X1, test_Y=test_Y)
