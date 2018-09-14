import argparse
import numpy as np
import json
import random
import keras
import tensorflow as tf
import sklearn.metrics
from scipy.stats import norm as dist_model
from keras import backend as K


def l2ac_predict(model, data, top_n, vote_n=1):
    test_X, test_X1=data['test_X0'], data['test_X1']
    y_pred=[]
    for ix in range(test_X1.shape[1]): #through all candidate classes
        if vote_n>1:
            n_pred=[]
            for jx in range(-vote_n, 0):
                n_pred.append(model.predict([test_X, test_X1[:,ix,jx].reshape(-1,1) ] ) )
            n_pred=np.concatenate(n_pred, 1)
        else:
            n_pred=model.predict([test_X, test_X1[:,ix,-top_n:] ] )
        y_pred.append( np.expand_dims(n_pred, 1) )
    y_pred=np.concatenate(y_pred, 1)
    y_pred=y_pred[:,:,-vote_n:].sum(-1)/float(vote_n)
    return y_pred

def doc_predict(model, data, model_type):
    if "mlp" in model_type:
        y_pred=model.predict(data['test_X0'])
    else:
        y_pred=model.predict(data['test_idx_X0']) 
    return y_pred
        
def doc_thres(model, data, model_type, scale = 2.):
    train_y = data['train_set_Y']
    if "mlp" in model_type:
        train_pred = model.predict(data['train_set_X'])
    else:
        train_pred = model.predict(data['train_set_idx_X'])
    #print train_y.shape, train_pred.shape 
    def fit(prob_pos_X):
        prob_pos = [p for p in prob_pos_X]+[2-p for p in prob_pos_X]
        pos_mu, pos_std = dist_model.fit(prob_pos)
        return pos_std
    stds = []
    for c in range(train_pred.shape[-1]):
        idx = [train_y == c]
        c_pred = train_pred[idx]
        c_prob = c_pred[:,c]
        std = fit(c_prob)
        stds.append(std)
    thres = [max(0.5, 1. - scale * std) for std in stds]
    return thres

def evaluate(y_true, y_pred, thres=0.5, rejection=False, mode="weighted"):
    if rejection:
        if isinstance(thres, list):
            reject_pred = []
            for p in y_pred:# loop every test prediction
                max_class = np.argmax(p)# predicted class
                max_value = np.max(p)# predicted probability           
                if max_value > thres[max_class]:
                    reject_pred.append(0)#predicted probability is greater than threshold, accept
                else:
                    reject_pred.append(1)#otherwise, reject
            y_pred=np.concatenate([y_pred, np.expand_dims(reject_pred, 1) ], 1) 
        else:
            y_pred=np.concatenate([y_pred, np.expand_dims(y_pred.max(axis=1)<=thres, 1) ], 1)
    else:
        keep_idx=(y_true!=y_true.max() )
        y_pred=y_pred[keep_idx]
        y_true=y_true[keep_idx]
    y_pred=y_pred.argmax(axis=1)
    return sklearn.metrics.f1_score(y_true, y_pred, average=mode), y_true, y_pred

def pred_evaluate(config):
    set_modes=config["set_modes"] #["test_25", "test_50", "test_75"]
    db=config["db"] #"amazon" 
    out_dir=config["out_dir"]
    model_type=config["model_type"]
    doc_eval="DOC" in model_type 
    if not doc_eval:
        top_n=config["top_n"] #10
        vote_n=config["vote_n"] #1 #typically 1, we disable manual vote; when top_n=1, we optionally vote

    scores={}
    for set_mode in set_modes:
        data=np.load("../"+db+"/data/"+set_mode+"_idx.npz")
        sess=tf.Session()
        K.set_session(sess)
        if doc_eval:
            model_fn=out_dir+"eval_"+set_mode+".h5"
            model=keras.models.load_model(model_fn)
        else:
            model_fn=out_dir+"eval.h5"
            model=keras.models.load_model(model_fn)
            model.get_layer("embedding_1").set_weights([np.vstack([data['train_rep'], np.zeros((90000, 512))]) ])
        
        thres=0.5
        if doc_eval:
            y_pred=doc_predict(model, data, model_type)
            gaus_thres=doc_thres(model, data, model_type)
        else:
            y_pred=l2ac_predict(model, data, top_n, vote_n)
            
        weighted_f1, _, _=evaluate(data['test_Y'], y_pred, thres=thres, rejection=True, mode="weighted")
        macro_f1, _, _=evaluate(data['test_Y'], y_pred, thres=thres, rejection=True, mode="macro")
        micro_f1, _, _=evaluate(data['test_Y'], y_pred, thres=thres, rejection=True, mode="micro")
        scores[set_mode]={'weighted_f1': weighted_f1, 'macro_f1': macro_f1, 'micro_f1': micro_f1}
        if doc_eval:
            weighted_f1, _, _=evaluate(data['test_Y'], y_pred, thres=gaus_thres, rejection=True, mode="weighted")
            macro_f1, _, _=evaluate(data['test_Y'], y_pred, thres=gaus_thres, rejection=True, mode="macro")
            micro_f1, _, _=evaluate(data['test_Y'], y_pred, thres=gaus_thres, rejection=True, mode="micro")
            scores[set_mode+"_gaus"]={'weighted_f1': weighted_f1, 'macro_f1': macro_f1, 'micro_f1': micro_f1}
            
        K.clear_session() 
    with open(out_dir+"eval.json", "w") as fw:
        json.dump(scores, fw)


parser = argparse.ArgumentParser(description="Evaluation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('config', type=str)

    
if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config) as f:
        config=json.load(f)
    pred_evaluate(config)
