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
sns.set()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# In[2]:


##create the keras model (LR in this case)
def create_LR_model():
    initializer = tf.keras.initializers.Constant(value=0)
    ##build LR model
    number_of_classes = 1
    number_of_features = 70
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(number_of_classes,activation = 'sigmoid',
                                    input_dim = number_of_features,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


# In[20]:


def clear_clients(path, numofhours):
    ##clear models from clients
    command = f'del /S C:{path}client/model/LR/*_{numofhours}*  {path}server/model/server_models/best_model/LR/*_{numofhours}* {path}server/model/server_models/current_model/LR/*_{numofhours}*' 
    command = command.replace('/',"\\")
    command = f'{command} /s /q'
    subprocess.call(command, shell = True)
    return


# In[18]:


def run_clients(client, path, epochs, test, numofhours, run):
    ##send model to clients
    if test:
        command = f'xcopy "{path}server/model/server_models/best_model/LR/client_{client}_{numofhours}_{run}.h5" "C:{path}client/model/LR/"'
    else:
        command = f'xcopy "{path}server/model/server_models/current_model/LR/client_{client}_{numofhours}_{run}.h5" "C:{path}client/model/LR/"'
    ##parse the copy command
    command = command.replace('/',"\\")
    command = f'{command} /e /s /y'
    subprocess.call(command, shell = True) 
    
    ##Run script
    command = f'python {path}client/clientServer-LR.py -cl={client} -ep={epochs} -ts={test} -hr={numofhours} -rn={run} -en=False'
    command = command.split(' ')
    output = subprocess.check_output(command) 
    server_response = output.decode('utf-8').split(' ')
    return server_response


# In[5]:


def fedAvg(models, relative_weights):
    params = [models[m].get_weights() for m in models ]
    new_weights = []
    for weights_list_tuple in zip(*params):
        new_weights.append(np.array([np.average(np.array(
            weights_), axis=0, weights=relative_weights) for weights_ in zip(*weights_list_tuple)]))
    return new_weights


# In[6]:


def validateResults(response):
    # Check if the current validation is better than the previous best one
    sizes = np.array([x[0] for x in response.values()]) 
    weights = sizes / np.sum(sizes)

    scores = np.array([x[4] for x in response.values()]) 
    current_auc = np.sum(scores*weights)
    
    return current_auc, scores, weights


# In[7]:


def processResponse(command):
        # Retrieve server responses and parse
        result = [x.replace('\r\n','').replace('copiedX_train_normX_test_normX_val_normy_trainy_testy_val','') for x in command.result()[2:]]
        server_1_response = [float(j) for j in result]
        return server_1_response


# In[1]:


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



     # Hyperparameters
    patience = 5
    epochs = 20

    # Delete past client data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(clear_clients, path)

    # Model architecture
    #LR
    for client in clients:
        federated_model_LR = create_LR_model()
        federated_model_LR.save(f'{path}server/model/server_models/current_model/LR/client_{client}_{numofhours}_{run}.h5', save_format = 'h5')
        federated_model_LR.save(f'{path}server/model/server_models/best_model/LR/client_{client}_{numofhours}_{run}.h5', save_format = 'h5')

    # Set runtime parameters
    patience_counter = 0
    iterations = 0
    highest_auc = 0
    early_stopping = False
    test = False
    while (early_stopping == False) and (iterations < 20):
        # Run model
        commands = {}
        response = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in clients:
                commands[client] = executor.submit(run_clients, client, path, epochs, test, numofhours, run)
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
            models[client] = tf.keras.models.load_model(f'{path}server/model/client_models/LR/client_{client}_{numofhours}_{run}.h5')

        #Replace best models if applicable
        if current_auc > highest_auc:
            patience_counter = 0
            for client in clients:
                models[client].save(f'{path}server/model/server_models/best_model/LR/client_{client}_{numofhours}_{run}.h5', save_format = 'h5')
                highest_auc = current_auc
            print(f'Validation AUC: {current_auc}')
        else:
            patience_counter += 1

        # Conduct federated averaging to update the federated_model if we have not exceeded patience
        if patience_counter > patience:
            early_stopping = True
            test = True
        else:

            new_weights = fedAvg(models, weights)
            for client in clients:
                models[client].set_weights(new_weights)
                models[client].save(f'{path}server/model/server_models/current_model/LR/client_{client}_{numofhours}_{run}.h5', save_format = 'h5')
        iterations +=1

        if iterations >= 20:
            test = True


    if test:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in clients:
                commands[client] = executor.submit(run_clients, client, path, epochs, test, numofhours, run)
        for client in clients:
            # Retrieve server responses
            response[client] = [float(x.replace('\r\n','').replace('X_train_normX_test_normX_val_normy_trainy_testy_val','')) for x in commands[client].result()]

        current_auc, client_auc, weights = validateResults(response)

        print(f'Test AUC: {current_auc}')

if __name__ == '__main__':
    main()

