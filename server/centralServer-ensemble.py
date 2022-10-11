#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")
import argparse
import concurrent.futures
import contextlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
import spur
import subprocess
import sys
import tensorflow as tf
import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


def create_data(clients, numofhours):
    for client in clients:
        command = f'python {path}client/clientServer-dataset.py -cl={client} -hr={numofhours}'
        command = command.split(' ')
        subprocess.check_output(command)


# In[3]:


def run_clients(path, clients, numofhours, run):
    models = ['LR','RF', 'SVM']
    ##Run script
    results = {}
    for model in models:
        command = f'python {path}server/centralServer-{model}.py -cl={clients} -hr={numofhours} --run={run}'
        command = command.split(' ')
        output = subprocess.check_output(command) 
        results[model] = output.decode('utf-8')
    return results


# In[4]:


def run_ensemble(path, clients, numofhours, run):
    ensemble_auc = []
    sample_size = []
    for client in clients:
        command = f'python {path}client/clientServer-Ensemble.py --client={client} -hr={numofhours} -rn={run}'
        command = command.split(' ')
        output = subprocess.check_output(command)
        output = [float(x) for x in output.decode('utf-8').split(',')]
        ensemble_auc.append(output[1])
        sample_size.append(output[0])
    return np.array(sample_size), np.array(ensemble_auc)


# In[5]:


def averageResults(sample_size, ensemble_auc):
    relative_weights = sample_size / sum(sample_size)
    return np.sum(relative_weights * ensemble_auc)


# In[6]:


clients = 'CUMC,UTH,Aachen'
path = '/Users/ae2722/Documents/DCI_code/'
clients_ = clients.split(',')
range_idx = np.flip(np.arange(0, 168, 12))
results, results_sites = {},[{},{},{}]


# In[10]:


range_idx = np.flip(np.arange(0, 168, 12))
for i in range_idx[:-2]:
    results[i] = []
    for client, res in zip(clients_, results_sites):
        res[i] = []


# In[ ]:


for _ in range(100):
    for numofhours in range_idx[:-2]:
        model_results = run_clients(path,clients, numofhours, i)
        sample_size, ensemble_auc = run_ensemble(path, clients_, numofhours)
        auc = averageResults(sample_size, ensemble_auc)
        print(f'Ensemble AUC: {auc}')
        results[numofhours].append(auc)
        #client specific results
        for client, res in zip(clients_, results_sites):
            sample_size_, ensemble_auc_ = run_ensemble(path, [client,client], numofhours)
            auc_ = averageResults(sample_size_, ensemble_auc_)
            res[numofhours].append(auc_)
            print(f'{client} AUC: {auc_}')
    ##save each time
    pd.DataFrame.from_dict(results, orient = 'index').to_csv('results_.csv')
    for client, res in zip(clients, results_sites):
        pd.DataFrame.from_dict(res, orient = 'index').to_csv(f'results_{client}.csv')
        


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s0','--centralServer', default = '')
    parser.add_argument('-cl','--clients', default = '')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clients = args.clients
    path = args.path
    
    clients_ = clients.split(',')
    range_idx = np.flip(np.arange(0, 168, 12))
    results = {}
    
    for numofhours in range_idx:
        create_data(clients, numofhours)
        results = run_clients(path,clients, numofhours)
        sample_size, ensemble_auc = run_ensemble(path, clients_, numofhours)
        auc = averageResults(sample_size, ensemble_auc)
        print(f'Ensemble AUC: {auc}')
        results[numofhours] = auc

if __name__ == '__main__':
    main()
    

