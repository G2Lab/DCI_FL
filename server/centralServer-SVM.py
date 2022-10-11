#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
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
import time
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import _pickle as cPickle
sns.set()


# In[38]:


def create_SVM_model(path, client, numofhours, run):
    ##0 weight vector for number of features
    model = SGDClassifier(loss = 'hinge', penalty = 'l2', max_iter =20)
    model = CalibratedClassifierCV(model)
    ##save in central
    with open(f'{path}server/model/server_models/current_model/SVM/SVM_{client}_{numofhours}_{run}.pkl', 'wb') as f:
        cPickle.dump(model, f)


# In[39]:


##load the rf  model
def save_SVM_model(path, client_model):
    with open(f'{path}', 'wb') as f:
        cPickle.dump(client_model, f)
    return


# In[40]:


def load_SVM_model(location):
    with open(location, 'rb') as f:
        return  cPickle.load(f)


# In[41]:


def clear_clients(path, numofhours, run):
    ##clear models from clients
    command = f'del C:{path}client/model/SVM/SVM_*_{numofhours}_{run}*'
    command = command.replace('/',"\\")
    command = f'{command} /s /q'
    subprocess.call(command, shell = True)
    return


# In[42]:


def run_clients(client, path, test, RBF, numofhours, run):
    ##send model to clients
    if test:
        command = f'xcopy "C:{path}server/model/server_models/best_model/SVM/SVM_{client}_{numofhours}_{run}.pkl" "C:{path}client/model/SVM/"'
    else:
        command = f'xcopy "C:{path}server/model/server_models/current_model/SVM/SVM_{client}_{numofhours}_{run}.pkl" "C:{path}client/model/SVM/"'
    ##parse the copy command
    command = command.replace('/',"\\")
    command = f'{command} /e /s /y'
    subprocess.call(command, shell = True) 
    
    
    ##Run script
    command = f'python {path}client/clientServer-SVM.py -cl={client} -ts={test} -rb={RBF} -en=False -hr={numofhours} -rn={run}'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[43]:


def exctract_coefs(client_model):
    coef_avg = 0
    for m in client_model.calibrated_classifiers_:
        coef_avg = coef_avg + m.base_estimator.coef_
    coef_avg  = coef_avg/len(client_model.calibrated_classifiers_)
    return coef_avg


# In[44]:


def fedAvg(models, relative_weights):
    #extract weights
    coefs = [exctract_coefs(models[model]) for model in models]
    #average weights
    weights_ = np.average(coefs, weights = relative_weights, axis = 0)
    # assign to the classifier
    for model in models:
        for m in models[model].calibrated_classifiers_:
            m.base_estimator.coef_ = weights_
    return models


# In[45]:


def validateResults(response):
    # Check if the current validation is better than the previous best one
    sizes = np.array([x[0] for x in response.values()]) 
    weights = sizes / np.sum(sizes)

    scores = np.array([x[2] for x in response.values()]) 
    current_auc = np.sum(scores*weights)
    
    return current_auc, scores, weights


# In[46]:


def processResponse(command):
        # Retrieve server responses and parse
        result = [x.replace('\r\n','').replace('copied','') for x in command.result()[2:]]
        server_1_response = [float(j) for j in result]
        return server_1_response


# In[ ]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s0','--centralServer', default = '')
    parser.add_argument('-cl','--clients')
    parser.add_argument('-rb','--RBF', default = 'False')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    
    args = parser.parse_args()
    
    centralServer = args.centralServer
    clients = args.clients
    RBF = args.RBF
    path = args.path
    numofhours = args.hours
    run=args.run
    
    clients = clients.split(',')
    
    # Delete past client data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(clear_clients, path, numofhours, run)

    # Model architecture
    for client in clients:
        model = create_SVM_model(path, client, numofhours, run)

    # Set runtime parameters
    test = False
    highest_auc = 0
    early_stopping = False
    iterations = 0
    patience_counter = 0
    patience = 5
    RBF = False

    # Run model
    while (early_stopping == False) and (iterations < 5):
        commands = {}
        response = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in clients:
                commands[client] = executor.submit(run_clients, client, path, test, RBF, numofhours, run)
        for client in clients:
            # Retrieve server responses
            response[client] = processResponse(commands[client])

        # Wait until the two servers return their weights files
        while sum([f'_{numofhours}_{run}' in x for x in  list(set(os.listdir(f'{path}server/model/client_models/LR')))]) != len(clients):
            time.sleep(5)
        current_auc, client_auc, weights = validateResults(response)
        print(current_auc, client_auc)




        # Load models
        models = {}
        for client in clients:
            models[client] = load_SVM_model(f'{path}server/model/client_models/SVM/SVM_{client}_{numofhours}_{run}.pkl')



        # Conduct federated averaging to update the federated_model
        models = fedAvg(models, weights)
        for client in clients:
            save_SVM_model(f'{path}server/model/server_models/current_model/SVM/SVM_{client}_{numofhours}_{run}.pkl',  models[client])


        #Replace best models if applicable
        if current_auc > highest_auc:
            patience_counter = 0
            for client in clients:
                save_SVM_model(f'{path}server/model/server_models/best_model/SVM/SVM_{client}_{numofhours}_{run}.pkl', models[client])
                highest_auc = current_auc
            print(f'Validation AUC: {current_auc}')
        else:
            patience_counter += 1
        iterations +=1

        if (patience_counter > patience) or (iterations >= 5):
            early_stopping = True
            test = True


    if test:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in clients:
                commands[client] = executor.submit(run_clients, client, path, test, RBF, numofhours, run)
        for client in clients:
            # Retrieve server responses
            response[client] = [float(x.replace('\r\n','')) for x in commands[client].result()]

        current_auc, client_auc, weights = validateResults(response)

        print(f'Test AUC: {current_auc}')


            
if __name__ == '__main__':
    main()

