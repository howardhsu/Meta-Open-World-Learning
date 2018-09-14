import argparse
import numpy as np
import json
import random
import keras
import tensorflow as tf
from keras import backend as K


def train(config):    
    batch_size=config['batch_size']
    top_n=config["top_n"] #10
    model_type=config['model_type']
    set_mode=config["set_mode"] #train1
    db=config["db"] #amazon
    ncls=config["ncls"] #6

    out_dir=config["out_dir"]

    output_fn=out_dir+"train.h5"
    eval_fn=out_dir+"eval.h5"
    data=np.load("../"+db+"/data/"+set_mode+"_idx.npz")

    train_X0=np.repeat(data['train_X0'], ncls, axis=0)
    train_X1=data['train_X1'][:,-ncls:,-top_n:].reshape(-1, top_n)
    train_Y=data['train_Y'][:,-ncls:].reshape(-1,)
    valid_X0=np.repeat(data['valid_X0'], 2, axis=0) #the validation data is balanced.
    valid_X1=data['valid_X1'][:,-2:,-top_n:].reshape(-1, top_n)
    valid_Y=data['valid_Y'][:,-2:].reshape(-1,)
        
    query=keras.Input(shape=(data['train_rep'].shape[1],), dtype="float32" )
    x_rep=query
    x1=keras.Input(shape=(top_n, data['train_rep'].shape[1]), dtype="float32" )
    x1_rep=x1
    
    #"many_lstm_simspace"
    x_rep=keras.layers.RepeatVector(train_X1.shape[1] )(x_rep)
    if "abssub" in model_type:
        x_sub=keras.layers.Subtract()([x_rep, x1_rep])
        x_rep=keras.layers.Lambda(lambda x: K.abs(x) )(x_sub)
    elif "sum" in model_type:
        x_rep=keras.layers.Add()([x_rep, x1_rep])
    else:
        x_sub=keras.layers.Subtract()([x_rep, x1_rep])
        x_abs=keras.layers.Lambda(lambda x: K.abs(x) )(x_sub)
        x_add=keras.layers.Add()([x_rep, x1_rep])
        x_rep=keras.layers.Concatenate(axis=-1)([x_abs, x_add])
    x_rep=keras.layers.Dense(512, activation="relu")(x_rep)
    x_rep=keras.layers.Dropout(0.5)(x_rep)
    if top_n>1: #do not use lstm when only have one example per class
        x_rep=keras.layers.Dense(1, activation="sigmoid")(x_rep)
        x_rep=keras.layers.Bidirectional(keras.layers.CuDNNLSTM(1) )(x_rep)
    else:
        x_rep=keras.layers.Reshape((-1,) )(x_rep)
    output=keras.layers.Dense(1, activation="sigmoid")(x_rep)
    matching_model=keras.engine.Model([query, x1], output)
    mem_layer=keras.layers.Embedding(data['train_rep'].shape[0], data['train_rep'].shape[1], weights=[data['train_rep'] ], trainable=False)
    x=keras.Input(shape=(train_X0.shape[1],), dtype="int32" )
    x_rep=keras.layers.Reshape((-1,) )(mem_layer(x) )
    x1=keras.Input(shape=(train_X1.shape[1],), dtype="int32" )
    x1_rep=mem_layer(x1)
    output=matching_model([x_rep, x1_rep])
    train_model=keras.engine.Model([x, x1], output)
    train_model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001), metrics=["acc"])
    
    history=train_model.fit([train_X0, train_X1 ], train_Y, class_weight={0: 1./ncls, 1: (ncls-1.)/ncls}, 
                  validation_data=([valid_X0, valid_X1], valid_Y), 
                  batch_size=batch_size, epochs=50, verbose=0,
                  callbacks=[
                     keras.callbacks.ModelCheckpoint(output_fn, save_best_only=True) ]  )
    
    
    train_model.load_weights(output_fn)
    output=matching_model([query, x1_rep])
    eval_model=keras.engine.Model([query, x1], output)
    eval_model.save(eval_fn)
            
parser = argparse.ArgumentParser(description="Train a model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('config', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config=json.load(f)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    tf.set_random_seed(config['seed'])
    
    train(config)