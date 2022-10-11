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
import ast


# In[2]:


path = '/Users/ae2722/Documents/DCI_code/'


# In[3]:


numofhours = 156
clients = 'CUMC,UTH,Aachen'.split(',')


# In[4]:


def feature_selection(path, clients, numofhours):
    feature_scores = np.zeros(134)
    for client in clients:
        command = f'python {path}client/clientServer-dataset.py --client={client} -hr={numofhours} --features=True'
        command = command.split(' ')
        output = subprocess.check_output(command)
        out = output.decode('utf-8')
        out_ = out.replace('[','').replace(']','').replace('\r\n','').split(' ')
        size = int(out_[0])
        scores = out_[1:]
        feature_scores += [float(x)*size for x in scores]
    features = np.argpartition(feature_scores,-70)[-70:]
    np.savetxt(f'{path}client/Data_subpopn/features_{numofhours}.csv', features)
    return


# In[5]:


def create_data(path, clients, numofhours):
    for client in clients:
        command = f'python {path}client/clientServer-dataset.py -cl={client} -hr={numofhours}'
        command = command.split(' ')
        subprocess.check_output(command)


# In[9]:


range_idx = np.flip(np.arange(0, 168, 12))
for numofhours in [164]:
    feature_selection(path, clients, numofhours)
    create_data(path, clients, numofhours)


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
        feature_selection(path, clients, numofhours)
        create_data(path, clients, numofhours)

if __name__ == '__main__':
    main()
    

