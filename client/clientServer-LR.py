#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
sys.path.append('C:\\Users\\ae2722\\Documents\\DCI_code\\helperCode')
import main_create_model as mm
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn.model_selection import train_test_split
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
import argparse


# In[82]:


##load the data and create train/test split
def dataloader(path, client, numofhours):
    dfs = ['X_train_norm', 'X_test_norm', 'X_val_norm', 'y_train', 'y_test', 'y_val']
    for i, df in zip(range(6), dfs):
        globals()[df] = pd.read_excel(open(f'{path}client/Data_subpopn/dataset_client_{client}_{numofhours}.xlsx', 'rb'),
                  sheet_name=f'sheet{i}', index_col = 0)
        print(df)
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[5]:


##create the keras model (LR in this case)
def create_keras_model():
    initializer = tf.keras.initializers.GlorotNormal(seed=0)
    ##build LR model
    number_of_classes = 1
    number_of_features = X_train_norm.shape[1]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(number_of_classes,activation = 'sigmoid',
                                    input_dim = number_of_features,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=0.1)))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    return model


# In[6]:


def run_model(path, client, test, ensemble, num_epochs, numofhours, run):
    # Import the data
    X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val = dataloader(path, client, numofhours)
    client_model = tf.keras.models.load_model(f'{path}client/model/LR/client_{client}_{numofhours}_{run}.h5')
    if test == 'False':
        # Evaluate on the validation set    
        validate_loss, validate_auc = 0, 0  #client_model.evaluate(X_val_norm, y_val, verbose=0)
        # Train model
        history = tf.keras.callbacks.History()
        with contextlib.redirect_stdout(None):
            client_model.fit(X_train_norm, y_train, verbose=0, epochs=num_epochs, callbacks=[history])

        # Save model
        client_model.save(f'{path}client/model/LR/client_{client}_{numofhours}_{run}.h5', save_format = 'h5')
        # Calculate loss + auc
        keys = list(history.history.keys())
        train_loss = history.history[keys[0]][-1]
        train_auc = history.history[keys[1]][-1]


        # Send the weights back to master server
        command = f'xcopy {path}client/model/LR/client_{client}_{numofhours}_{run}.h5 {path}server/model/client_models/LR/'
        command = command.replace('/',"\\")
        command = f'{command} /e /s /y'
        subprocess.call(command, shell = True) 

        # Reset stdout and print
        print(len(X_train_norm), train_loss, train_auc, train_loss, train_auc)

    elif ensemble == 'True':
        client_model = tf.keras.models.load_model(f'{path}server/model/server_models/best_model/LR/client_{client}_{numofhours}_{run}.h5')
        predicted = client_model.predict(X_test_norm, verbose=0)
        print(len(X_test_norm), predicted.reshape(1,-1))

    else:
        client_model = tf.keras.models.load_model(f'{path}server/model/server_models/best_model/LR/client_{client}_{numofhours}_{run}.h5')
        test_loss, test_auc = client_model.evaluate(X_test_norm, y_test, verbose=0)
        print(len(X_test_norm), 0, 0, test_loss, test_auc)
    


# In[ ]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-ep','--epochs', default = '20')
    parser.add_argument('-ts','--test')
    parser.add_argument('-en','--ensemble')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-rn','--run')
    
    args = parser.parse_args()
    
    client = args.client
    num_epochs = int(args.epochs)
    test = args.test
    ensemble = args.ensemble
    path = args.path
    numofhours = int(args.hours)
    run = args.run

    
    run_model(path, client, test, ensemble, num_epochs, numofhours, run)
        
if __name__ == '__main__':
    main()

