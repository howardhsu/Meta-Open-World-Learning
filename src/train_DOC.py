import argparse
import numpy as np
import json
import random
import keras
import tensorflow as tf
from keras import backend as K


def train(config):
    batch_size=config['batch_size']
    model_type=config['model_type'] #["mlp_256"] #, "lstm_512"]
    set_modes=config["set_modes"] #["test_25", "test_50", "test_75"]
    db=config["db"] # "amazon"
    emb=config["emb"] #"../amazon/data/embedding.npy"
    out_dir=config["out_dir"]
    
    for set_mode in set_modes:
        eval_fn=out_dir+"eval_"+set_mode+".h5"
        data=np.load("../"+db+"/data/"+set_mode+"_idx.npz")
        if 'mlp' in model_type:
            train_X=data['train_set_X']
        else:
            train_X=data['train_set_idx_X']
            glove_pre_trained=np.load(emb)
        train_Y=np.zeros((data['train_set_Y'].shape[0], data['train_set_Y'].max()+1), dtype='int16')
        train_Y[np.arange(train_Y.shape[0]), data['train_set_Y'] ]=1

        sess=tf.Session()
        K.set_session(sess)
        if 'mlp' in model_type:
            x=keras.Input(shape=(data['train_set_X'].shape[1],), dtype="float32" )
            x_rep=x
            enc=keras.layers.Dense(256, activation="relu")(x_rep)
        else:
            x=keras.Input(shape=(data['train_set_idx_X'].shape[1],), dtype="int32" )
            emb_layer=keras.layers.Embedding(glove_pre_trained.shape[0], glove_pre_trained.shape[1], weights=[glove_pre_trained], trainable=False)
            x_rep=emb_layer(x)
        if "lstm_512" in model_type:
            x_rep=keras.layers.Dropout(0.5)(x_rep)
            lstm_layer2=keras.layers.Bidirectional(keras.layers.CuDNNLSTM(256) )
            enc=keras.layers.Activation('relu')(lstm_layer2(x_rep) )
        elif "cnn" in model_type:
            graph_in = keras.Input(shape=(data['train_set_idx_X'].shape[1],  glove_pre_trained.shape[1]))
            convs = []
            for fsz in [3, 4, 5]:
                conv = keras.layers.Conv1D(filters=128,
                                         kernel_size=fsz,
                                         padding='valid',
                                         activation='relu')(graph_in)
                pool = keras.layers.GlobalMaxPooling1D()(conv)
                convs.append(pool)
            out = keras.layers.Concatenate(axis=-1)(convs)
            graph = keras.models.Model(inputs=graph_in, outputs=out) #convolution layers

            x_rep=keras.layers.Dropout(0.5)(x_rep)
            x_conv = graph(x_rep)
            x_conv = keras.layers.Dropout(0.5)(x_conv)
            enc = keras.layers.Dense(256, activation="relu")(x_conv)

        x_rep=keras.layers.Dropout(0.5)(enc)
        output=keras.layers.Dense(data['train_set_Y'].max()+1, activation="sigmoid")(x_rep)
        model=keras.engine.Model(x, output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
        history=model.fit(train_X, train_Y, 
                          validation_split=0.1, 
                          batch_size=batch_size, epochs=200, verbose=0,
                          callbacks=[
                             keras.callbacks.ModelCheckpoint(eval_fn, save_best_only=True) ]  )
        K.clear_session()

            
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