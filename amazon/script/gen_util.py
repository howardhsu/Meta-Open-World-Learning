import os
import random
import numpy as np
import nltk
import sklearn
import keras
import scipy

#class set sampling
def gen_run_sets(run, classes, group_sizes):
    run_sets=[]
    for _ in range(run):
        class_sets=[]
        prev_set=classes
        for g in group_sizes:
            class_sets.append(random.sample(prev_set, g) )
            prev_set=class_sets[-1]
        run_sets.append(class_sets[::-1])
    return run_sets

def data2np_train_idx_neg_cls(class_set, data_rep, data_cls_rep, classes, train_per_cls, top_k): 
    X0, X1, Y=[], [], []
    base_cls_offset=classes.index(class_set[0])
    for cls in class_set:
        tmp_X1=[]
        tmp_Y=[]
        ix=classes.index(cls) #considers validation_set that not start from zero.
        cls_offset=ix*train_per_cls
        
        #find top_k non similar classes.
        rest_cls_idx=[classes.index(cls1) for cls1 in class_set if classes.index(cls1)!=ix]
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset+train_per_cls], data_cls_rep[rest_cls_idx] )
        sim_idx=sim.argsort(axis=1)
        sim_idx+=base_cls_offset
        sim_idx[sim_idx>=ix]+=1
        sim_idx=sim_idx[:,-top_k:]

        for kx in range(-top_k, 0):
            tmp_X1_batch=[]
            for jx in range(train_per_cls):
                cls1_offset=sim_idx[jx, kx]*train_per_cls
                sim1_idx=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset+jx:cls_offset+jx+1], data_rep[cls1_offset:cls1_offset+train_per_cls] )
                sim1_idx=sim1_idx.argsort(axis=1)[:1,-(train_per_cls-1):]
                sim1_idx+=cls1_offset
                tmp_X1_batch.append(np.expand_dims(sim1_idx, 1) )
            tmp_X1_batch=np.vstack(tmp_X1_batch)
            tmp_X1.append(tmp_X1_batch)    
            tmp_Y.append(np.full((train_per_cls, 1), 0) )
        
        #put sim in the last dim
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset+train_per_cls])
        sim_idx=sim.argsort(axis=1)[:,:-1]+cls_offset #add the offset to obtain the real offset in memory.
        tmp_X1.append(np.expand_dims(sim_idx, 1) )
        tmp_Y.append(np.full((train_per_cls, 1), 1) )
        
        X0.append(np.arange(cls_offset, cls_offset+train_per_cls ).reshape(-1, 1) )
        X1.append(np.concatenate(tmp_X1, 1) )
        Y.append( np.concatenate(tmp_Y, axis=1)  ) #similar

    X0=np.vstack(X0)
    X1=np.vstack(X1)
    Y=np.concatenate(Y)
    shuffle_idx=np.random.permutation(X0.shape[0])
    return X0[shuffle_idx], X1[shuffle_idx], Y[shuffle_idx]


#generate pairwise data for a run sample on a data split.
def data2np_train_idx(class_set, data_rep, classes, train_per_cls): 
    # dont need to sample since we want to do top-any
    # note you always have top-(train_per_cls-1) because you can't make similar to yourself.
    X0, X1, Y=[], [], []
    base_offset=(classes.index(class_set[0] )*train_per_cls)
    for cls in class_set:
        tmp_X1=[]
        ix=classes.index(cls) #considers validation_set that not start from zero.
        cls_offset=ix*train_per_cls
        #in-class similarity
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset+train_per_cls])
        #sim=sim.argsort(axis=1)
        #sim_idx=np.zeros((train_per_cls, train_per_cls-1), 'int32')
        #for ix in range(train_per_cls):
        #    jx_idx=0
        #    for jx in range(train_per_cls):
        #        if sim[ix,jx]!=ix:
        #            sim_idx[ix,jx_idx]=sim[ix,jx]
        #            jx_idx+=1
        sim_idx=sim.argsort(axis=1)[:,:-1]+cls_offset #add the offset to obtain the real offset in memory.
        tmp_X1.append(np.expand_dims(sim_idx, 1) )
        tmp_rep=[]
        for cls1 in class_set:
            jx=classes.index(cls1)
            if jx!=ix: #other class that are in current class set
                cls1_offset=jx*train_per_cls #this class offset to retrieve the examples.
                tmp_rep.append(data_rep[cls1_offset:cls1_offset+train_per_cls ])
        tmp_rep=np.vstack(tmp_rep)
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls_offset:cls_offset+train_per_cls], tmp_rep)
        sim_idx=sim.argsort(axis=1)[:,-(train_per_cls-1):]
        #total shift on class_set offset.
        sim_idx+=base_offset # the classes before the current class still need to add.
        sim_idx[sim_idx>=cls_offset]+=train_per_cls # jump to the offset after the current class.
        tmp_X1.append(np.expand_dims(sim_idx, 1) )
        
        X0.append(np.arange(cls_offset, cls_offset+train_per_cls ).reshape(-1, 1) )
        X1.append(np.concatenate(tmp_X1, 1) )
        Y.append(np.concatenate([np.full((train_per_cls, 1), 1), 
                                 np.full((train_per_cls, 1), 0) ], axis=1) ) #similar

    X0=np.vstack(X0)
    X1=np.vstack(X1)
    Y=np.concatenate(Y)
    shuffle_idx=np.random.permutation(X0.shape[0])
    return X0[shuffle_idx], X1[shuffle_idx], Y[shuffle_idx]


#generate pairwise data for a run sample on a data split.
def data2np_train(class_set, data_rep, sample, classes):
    X0=[]
    X1=[]
    Y=[]

    for cls in class_set:
        X0.append(data_rep[cls])
        tmp_X1=[]
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls])
        sim_idx=sim.argsort(axis=1)[:,-(sample+1):-1] #can't sample itself.
        #generate similar examples
        tmp_X1_nn=[]
        for jx in range(sim_idx.shape[1]): #from least simililar to most similar
            tmp_X1_nn.append( np.expand_dims(data_rep[cls][sim_idx[:, jx] ], 1) )
        tmp_X1.append(np.expand_dims(np.concatenate(tmp_X1_nn, 1), 1) )

        tmp_rep=[]
        for cls1 in class_set:
            if cls1!=cls:
                tmp_rep.append(data_rep[cls1])
        tmp_rep=np.vstack(tmp_rep)
        sim=sklearn.metrics.pairwise.cosine_similarity(data_rep[cls], tmp_rep)
        sim_idx=sim.argsort(axis=1)[:,-sample:]
        tmp_X1_nn=[]
        for jx in range(sim_idx.shape[1]):
            tmp_X1_nn.append( np.expand_dims(tmp_rep[sim_idx[:, jx] ], 1) )
        tmp_X1.append(np.expand_dims(np.concatenate(tmp_X1_nn, 1), 1) )
        X1.append(np.concatenate(tmp_X1, 1) )

        Y.append(np.concatenate([np.full((len(data_rep[cls]), 1), 1), 
                                 np.full((len(data_rep[cls]), 1), 0) ], axis=1) ) #similar

    X0=np.vstack(X0)
    X1=np.vstack(X1)
    Y=np.concatenate(Y)
    shuffle_idx=np.random.permutation(X0.shape[0])
    return X0[shuffle_idx], X1[shuffle_idx], Y[shuffle_idx]


def embedding_np(word_idx, output, path="../../lifelong_embedding/shared_data/glove.840B.300d.vec"):
    #build the glove embedding np
    embedding=np.zeros((len(word_idx)+2, 300) )
    with open(path) as f:
        for l in f:
            rec=l.rstrip().decode("utf-8").split(' ')
            assert len(rec)==301 or len(rec)==2
            if rec[0] in word_idx:
                embedding[word_idx[rec[0]]]=np.array([float(r) for r in rec[1:] ] ) 
    np.save(output, embedding.astype('float32') )