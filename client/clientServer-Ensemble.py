#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
import warnings
warnings.filterwarnings("ignore")
import contextlib
import math
import pickle as pkl
import random
import spur
import sys
sys.path.append('C:\\Users\\ae2722\\Documents\\DCI_code\\helperCode')
import main_create_model as mm
import argparse
import subprocess
from sklearn import metrics


# In[3]:


def dataloader(path, client, numofhours):
    dfs = ['X_train_norm', 'X_test_norm', 'X_val_norm', 'y_train', 'y_test', 'y_val']
    for i, df in zip(range(6), dfs):
        globals()[df] = pd.read_excel(open(f'{path}client/Data_subpopn/dataset_client_{client}_{numofhours}.xlsx', 'rb'),
                  sheet_name=f'sheet{i}', index_col = 0)
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[4]:


def test_probability_clients(path, client, numofhours, run):
    models = ['LR', 'RF', 'SVM']
    results = {}
    for model in models:
        command = f'python {path}client/clientServer-{model}.py -cl={client} -ts=Pass -en=True -hr={numofhours} -rn={run}'
        command = command.split(' ')
        output = subprocess.check_output(command) 
        output =  output.decode('utf-8').split('\n')
        results[model]= prob_parser(output)
    return results


# In[22]:


def prob_parser(output):
    output = [out for out in output if '_' not in out]
    parsed = []
    for i in output:
        parsed.extend([float(x) for x in i.replace('[', '').replace(']','').replace('\r','').split(' ') if x != ''])
    sample_size = parsed[0]
    preds = parsed[1:]
    return sample_size, preds


# In[37]:


def ensemble_learner(results):
    ensemble = []
    for _, preds in results.values():
        ensemble.append(preds)
    ensemble = np.array(ensemble)
    ensemble_mean = np.mean(ensemble, axis = 0)
    return ensemble_mean


# In[ ]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    args = parser.parse_args()
    
    client = args.client
    path = args.path
    numofhours = int(args.hours)
    run = args.run
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(path, client, numofhours)
    
    results = test_probability_clients(path, client, numofhours, run)
    ensemble_mean = ensemble_learner(results)
    fpr, tpr, thresholds = metrics.roc_curve(y_test.values, ensemble_mean)
    ensemble_auc = metrics.auc(fpr, tpr)
    print(f'{y_test.shape[0]}, {ensemble_auc}')

if __name__ == '__main__':
    main()

