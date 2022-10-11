#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings("ignore")
import contextlib
import math
import _pickle as cPickle
import random
import sys
sys.path.append('C:\\Users\\ae2722\\Documents\\DCI_code\\helperCode')
import main_create_model as mm
import argparse
import subprocess


# In[3]:


##load the data and create train/test split
def dataloader(path, client, numofhours):
    dfs = ['X_train_norm', 'X_test_norm', 'X_val_norm', 'y_train', 'y_test', 'y_val']
    for i, df in zip(range(6), dfs):
        globals()[df] = pd.read_excel(open(f'{path}client/Data_subpopn/dataset_client_{client}_{numofhours}.xlsx', 'rb'),
                  sheet_name=f'sheet{i}', index_col = 0)
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[4]:


##load the rf  model
def load_SVM_model(path, client, numofhours, run):
    with open(f'{path}client/model/SVM/SVM_{client}_{numofhours}_{run}.pkl', 'rb') as f:
        return  cPickle.load(f)


# In[5]:


##load the rf  model
def save_SVM_model(path, client, client_model, numofhours, run):
    with open(f'{path}client/model/SVM/SVM_{client}_{numofhours}_{run}.pkl', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[37]:


def run_model(path, client, RBF, test, ensemble, numofhours, run):
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(path, client, numofhours)
    # Load and run the neural network
    client_model= load_SVM_model(path, client, numofhours, run)
    if test == 'False':
        # Fit on the train data
        client_model.fit(X_train_norm, y_train)
        y_train_pred = client_model.predict_proba(X_train_norm)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_train, y_train_pred)
        train_auc = metrics.auc(fpr, tpr)
        # Evaluate on the validation set
        #y_val_pred = client_model.predict_proba(X_val_norm)[:,1]
        #fpr, tpr, thresholds = metrics.roc_curve(y_val, y_val_pred)
        val_auc = 0  #metrics.auc(fpr, tpr)
        # Save model
        save_SVM_model(path, client, client_model, numofhours, run)
        # Send the weights back to master server
        command = f'xcopy {path}client/model/SVM/SVM_{client}_{numofhours}_{run}.pkl {path}server/model/client_models/SVM/'
        command = command.replace('/',"\\")
        command = f'{command} /e /s /y'
        subprocess.call(command, shell = True)

        # Reset stdout and print
        print(len(X_train_norm), train_auc, train_auc)

    elif ensemble == 'True':
        with open(f'{path}server/model/server_models/best_model/SVM/SVM_{client}_{numofhours}_{run}.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        client_model = CalibratedClassifierCV(client_model)
        client_model.fit(X_train_norm, y_train)
        prediction = client_model.predict_proba(X_test_norm)[:,1]
        print(len(X_test_norm), prediction) 
        
    else:
        with open(f'{path}server/model/server_models/best_model/SVM/SVM_{client}_{numofhours}_{run}.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        y_test_pred = client_model.predict_proba(X_test_norm)[:,1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_pred)
        test_auc = metrics.auc(fpr, tpr)
        print(len(X_test_norm),0 , test_auc) 


# In[1]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-ts','--test')
    parser.add_argument('-rb','--RBF', default = 'False')
    parser.add_argument('-en','--ensemble')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    
    args = parser.parse_args()
    
    client = args.client
    test = args.test
    RBF = args.RBF
    ensemble = args.ensemble
    path = args.path
    numofhours = int(args.hours)
    run = args.run
    
    run_model(path, client, RBF, test, ensemble, numofhours, run)
        
if __name__ == '__main__':
    main()

