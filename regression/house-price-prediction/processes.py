# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 23:52:22 2020

@author: arun
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset,TensorDataset
from utils import *


##### Global variables 
net = torch.nn.Sequential(
        torch.nn.Linear(1, 200),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(200, 100),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(100, 1),
    )

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss



def train(args,labeled, resume_from, ckpt_file):
    trainDF = pd.read_csv(os.path.join(args["DATA_DIR"],'train.csv'))
    testDF = pd.read_csv(os.path.join(args["DATA_DIR"],'test.csv'))
    
    X , y, X_test = preprocess(trainDF,testDF)                                 ##### Note X_test here doesnot contain target hence using part of train as test
    
    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    splitix = int(args["VAL_PERCENT"] *len(X))
    X_train = X[:splitix,:]
    y_train = y[:splitix,:]
    traindataset = TensorDataset(X_train, y_train)
    train = Subset(traindataset,labeled)
    trainLoader = Data.DataLoader(dataset=train, 
                             batch_size=args["BATCH_SIZE"], 
                             shuffle=True, 
                             num_workers=2,)
    
    n_train =len(train)
    
    epochs = args["train_epochs"]
    for epoch in range(epochs):
        epoch_loss = 0
        epochstep = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for step, (batch_x, batch_y) in enumerate(trainLoader):
                epochstep+=1
                prediction = net(batch_x)
                loss = loss_func(prediction, b_y)
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()   
                loss.backward()         
                optimizer.step()
                print("epoch {} loss: {:.4f} avgloss: {:.4f}".format(epoch + 1, loss.item() , (epoch_loss/epochstep)))
        
        if not epoch%args["SAVE_PER_EPOCH"]:
            try:
                os.mkdir(args["EXPT_DIR"])
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                               args["EXPT_DIR"] + ckpt_file)
            logging.info(f'Checkpoint {epoch + 1} saved !')
    
    return args['OUTPUT_DIRECTORY']


def test(args, ckpt_file):
    
    trainDF = pd.read_csv(os.path.join(args["DATA_DIR"],'train.csv'))
    testDF = pd.read_csv(os.path.join(args["DATA_DIR"],'test.csv'))
    
    X , y, X_test = preprocess(trainDF,testDF)                                 ##### Note X_test here doesnot contain target hence using part of train as test
    
    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    splitix = int(args["VAL_PERCENT"] *len(X))
    X_train = X[splitix:,:]
    y_train = y[splitix:,:]
    testdataset = TensorDataset(X_train, y_train)
    testLoader = Data.DataLoader(dataset=testdataset, 
                             batch_size=args["BATCH_SIZE"], 
                             shuffle=False, 
                             num_workers=2,)
    
    net = torch.nn.Sequential(torch.nn.Linear(1, 200),
                              torch.nn.LeakyReLU(),
                              torch.nn.Linear(200, 100),
                              torch.nn.LeakyReLU(),
                              torch.nn.Linear(100, 1),
                            )
    
    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"] + ckpt_file)))
    net.eval()
    predix = 0
    predictions = {}
    truelabels = {}
    
    n_val = len(X_train)
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(testLoader):
            with torch.no_grad():
                prediction = net(batch_x)
                
            for logit,label  in zip(prediction,batch_y):
                predictions[predix] = logit.cpu().numpy().tolist()
                truelabels[predix] = label.cpu().numpy().tolist()
                predix+=1
            
            pbar.update()
        
    return {"predictions": predictions, "labels": truelabels}
        


def infer(args,unlabeled, ckpt_file):
    
    # Load the last best model
    trainDF = pd.read_csv(os.path.join(args["DATA_DIR"],'train.csv'))
    testDF = pd.read_csv(os.path.join(args["DATA_DIR"],'test.csv'))
    
    X , y, X_test = preprocess(trainDF,testDF)                                 ##### Note X_test here doesnot contain target hence using part of train as test
    
    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    splitix = int(args["VAL_PERCENT"] *len(X))
    X_train = X[:splitix,:]
    y_train = y[:splitix,:]
    traindataset = TensorDataset(X_train, y_train)
    train = Subset(traindataset,unlabeled)
    trainLoader = Data.DataLoader(dataset=train, 
                             batch_size=args["BATCH_SIZE"], 
                             shuffle=True, 
                             num_workers=2,)
    
    net = torch.nn.Sequential(torch.nn.Linear(1, 200),
                              torch.nn.LeakyReLU(),
                              torch.nn.Linear(200, 100),
                              torch.nn.LeakyReLU(),
                              torch.nn.Linear(100, 1),
                            )
    
    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join(args["EXPT_DIR"] + ckpt_file)))
    net.eval()
    predix = 0
    predictions = {}
    
    n_val = len(X_train)
    with tqdm(total=n_val, desc='Inference round', unit='batch', leave=False) as pbar:
        for step, (batch_x, batch_y) in enumerate(trainLoader):
            with torch.no_grad():
                prediction = net(batch_x)
                
            for logit in prediction:
                predictions[predix] = logit.cpu().numpy().tolist()
                predix+=1
            
            pbar.update()
    
    return {"outputs": predictions}
    

def getdatasetstate(args, split="train"):
    # Load the last best model
        trainDF = pd.read_csv(os.path.join(args["DATA_DIR"],'train.csv'))
        testDF = pd.read_csv(os.path.join(args["DATA_DIR"],'test.csv'))
    
        X , y, X_test = preprocess(trainDF,testDF)                                 ##### Note X_test here doesnot contain target hence using part of train as test
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)
    if split == "train":
        splitix = int(args["VAL_PERCENT"] *len(X))
        X_train = X[:splitix,:]
        y_train = y[:splitix,:]
    else:
        splitix = int(args["VAL_PERCENT"] *len(X))
        X_train = X[splitix:,:]
        y_train = y[splitix:,:]
    
    referencedict = {}
    for ix, row in y_train.iterrrows():
        referencedict[ix] = row                                ######### here index carries the necessary reference    
        
    return referencedict




if __name__ == "__main__":
    labeled = list(range(1000))
    resume_from = None
    ckpt_file = "ckpt_0"

    train(labeled=labeled, resume_from=resume_from, ckpt_file=ckpt_file)
    test(ckpt_file=ckpt_file)
    infer(unlabeled=[10, 20, 30], ckpt_file=ckpt_file)
