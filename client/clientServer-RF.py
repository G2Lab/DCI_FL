#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import contextlib
import math
import _pickle as cPickle
import random
import spur
import sys
sys.path.append('C:\\Users\\ae2722\\Documents\\DCI_code\\helperCode')
import main_create_model as mm
import argparse
import subprocess
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# In[2]:


##load the data and create train/test split
def dataloader(path, client, numofhours):
    dfs = ['X_train_norm', 'X_test_norm', 'X_val_norm', 'y_train', 'y_test', 'y_val']
    for i, df in zip(range(6), dfs):
        globals()[df] = pd.read_excel(open(f'{path}client/Data_subpopn/dataset_client_{client}_{numofhours}.xlsx', 'rb'),
                  sheet_name=f'sheet{i}', index_col = 0)
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[3]:


##load the rf  model
def load_RF_model(path, client, numofhours, run):
    with open(f'{path}client/model/RF/RF_{client}_{numofhours}_{run}.pkl', 'rb') as f:
        return  cPickle.load(f)


# In[4]:


##load the rf  model
def save_RF_model(path, client, client_model, numofhours, run):
    with open(f'{path}client/model/RF/RF_{client}_{numofhours}_{run}.pkl', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[25]:


def run_model(path, client, test, ensemble, numofhours, run):
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val =  dataloader(path, client, numofhours)
    # Load and run the neural network
    client_model = load_RF_model(path, client, numofhours, run)

    if test == 'False':
        # Run a param search
        param_grid = {
        'n_estimators': [5, 10, 20],
        'max_features': ['auto','log2'],
        "bootstrap": [True, False],
        'criterion': ["gini", "entropy"],
        'max_depth':[1,2,3,4,5,6,7,8,10,15,20]}

        n_iter_search = 16
        CV_rfc = RandomizedSearchCV(client_model, param_distributions=param_grid,
                                           n_iter=n_iter_search, cv=5,n_jobs=-1,scoring='roc_auc')
        CV_rfc.fit(X_train_norm, y_train)
        client_model = CV_rfc.best_estimator_
        
        fpr, tpr, thresholds = metrics.roc_curve(y_train.values, client_model.predict_proba(X_train_norm)[:,1])
        train_auc = metrics.auc(fpr, tpr)
        # Evaluate on the validation set
        #fpr, tpr, thresholds = metrics.roc_curve(y_val.values, client_model.predict_proba(X_val_norm)[:,1])
        valid_auc = 0 #metrics.auc(fpr, tpr)
        # Save model
        save_RF_model(path, client, client_model, numofhours, run)
        # Send the weights back to master server
        command = f'xcopy C:{path}client/model/RF/RF_{client}_{numofhours}_{run}.pkl C:{path}server/model/client_models/RF/'
        command = command.replace('/',"\\")
        command = f'{command} /e /s /y'
        subprocess.call(command, shell = True)

        # Reset stdout and print
        print(len(X_train_norm), train_auc, train_auc)
    
    elif ensemble == 'True':
        with open(f'{path}server/model/server_models/current_model/RF/RF_{client}_{numofhours}_{run}.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        predictions = client_model.predict_proba(X_test_norm)[:,1]
        print(len(X_test_norm), predictions)
    
    else:
        with open(f'{path}server/model/server_models/current_model/RF/RF_{client}_{numofhours}_{run}.pkl', 'rb') as f:
            client_model = cPickle.load(f)
        fpr, tpr, thresholds = metrics.roc_curve(y_test.values, client_model.predict_proba(X_test_norm)[:,1])
        test_auc = metrics.auc(fpr, tpr)
        print(len(X_test_norm),0 , test_auc)
        


# In[ ]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-ts','--test')
    parser.add_argument('-en','--ensemble')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    
    args = parser.parse_args()
    
    client = args.client
    test = args.test
    ensemble = args.ensemble
    path = args.path
    numofhours = int(args.hours)
    run = args.run
    
    run_model(path, client, test, ensemble, numofhours, run)
  
        
if __name__ == '__main__':
    main()

