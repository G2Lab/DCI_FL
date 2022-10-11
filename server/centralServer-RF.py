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
import _pickle as cPickle
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


##create the keras model RF in this case)
def create_RF_model(path, client, numofhours, run):
    model = RandomForestClassifier(max_depth=5, max_features = 'log2', criterion = 'gini', random_state=0)
    with open(f'{path}server/model/server_models/current_model/RF/RF_{client}_{numofhours}_{run}.pkl', 'wb') as f:
        cPickle.dump(model, f)
    return model


# In[3]:


##load the rf  model
def save_RF_model(location, client_model):
    with open(f'{location}.pkl', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[4]:


##load the rf  model
def load_RF_model(location):
    with open(f'{location}.pkl', 'rb') as f:
        return  cPickle.load(f)


# In[5]:


def clear_clients(client, path, numofhours):
    ##clear models from clients
    command = f'del C:{path}client/model/RF/RF_{client}_{numofhours}*.pkl'
    command = command.replace('/',"\\")
    command = f'{command} /s /q'
    return


# In[6]:


def run_clients(client, path, test, numofhours, run):
    ##send model to clients
    command = f'xcopy "C:{path}server/model/server_models/current_model/RF/RF_{client}_{numofhours}_{run}.pkl" "C:{path}client/model/RF/"'
    ##parse the copy command
    command = command.replace('/',"\\")
    command = f'{command} /e /s /y'
    subprocess.call(command, shell = True) 
    
    ##Run script
    command = f'python {path}client/clientServer-RF.py -cl={client} -ts={test} -en=False -hr={numofhours} -rn={run}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[7]:


def fedAvg(models, relative_weights):
    ##extract individual trees
    client_trees = [models[model].estimators_ for model in models]
    ##assign number of trees
    num_trees = [round(len(client_trees[0])*weight) for weight in relative_weights]
    ##sample from both forests
    new_forest = []
    np.random.seed(2)
    for i in range(len(client_trees)):
        new_forest.extend(list(np.random.choice(client_trees[i], num_trees[i])))
    ##assign to models
    for model in models:
        models[model].estimators_ = new_forest
    return models


# In[8]:


def validateResults(response):
    # Check if the current validation is better than the previous best one
    sizes = np.array([x[0] for x in response.values()]) 
    weights = sizes / np.sum(sizes)

    scores = np.array([x[2] for x in response.values()]) 
    current_auc = np.sum(scores*weights)
    
    return current_auc, scores, weights


# In[9]:


def processResponse(command):
        # Retrieve server responses and parse
        result = [x.replace('\r\n','').replace('copied','') for x in command.result()[2:]]
        server_1_response = [float(j) for j in result]
        return server_1_response


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s0','--centralServer', default = '')
    parser.add_argument('-cl','--clients', default = '')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clients = args.clients
    path = args.path
    numofhours = args.hours
    run = args.run
    
    clients = clients.split(',')
    
    # Delete past client data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(clear_clients, path, numofhours)

    # Model architecture
    for client in clients:
        model = create_RF_model(path, client, numofhours, run)
    # Set runtime parameters
    test = False

    # Run model
    commands = {}
    response = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for client in clients:
            commands[client] = executor.submit(run_clients, client, path, test, numofhours, run)
    for client in clients:
        # Retrieve server responses
        response[client] = processResponse(commands[client])

    # Wait until the two servers return their weights files
    while sum([f'_{numofhours}_{run}' in x for x in  list(set(os.listdir(f'{path}server/model/client_models/RF')))]) != len(clients):
        time.sleep(5)
    current_auc, client_auc, weights = validateResults(response)
    print(current_auc, client_auc)


    # Load models
    models = {}
    for client in clients:
        models[client] = load_RF_model(f'{path}server/model/client_models/RF/RF_{client}_{numofhours}_{run}')


    # Conduct federated averaging to update the federated_model
    models = fedAvg(models, weights)
    for client in clients:
        save_RF_model(f'{path}server/model/server_models/current_model/RF/RF_{client}_{numofhours}_{run}', models[client])


    test = True

    if test:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in clients:
                commands[client] = executor.submit(run_clients, client, path, test, numofhours, run)
        for client in clients:
            # Retrieve server responses
            response[client] = [float(x.replace('\r\n','')) for x in commands[client].result()]

        current_auc, client_auc, weights = validateResults(response)

        print(f'Test AUC: {current_auc}')


if __name__ == '__main__':
    main()

