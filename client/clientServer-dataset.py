#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper
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


# In[88]:


def feature_selection(client, path, numofhours):
    ##load the data and create train/test split
    keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']
    target = 'label'
    IDcol = 'Shopid'
    hourscol = 'hours'
    start_time = 24*3
    k_feat = 70

    all_predictors, labelsvals, demfeats_vals = mm.load_data_all(univ_type=client,include_dems = True)
        
    predictor_keys = mm.get_predictor(keys)
    demo_predictor_keys = mm.get_dem_predictors() 

    continuous_cols = np.array(predictor_keys)
    continuous_cols = np.append(continuous_cols, np.array(demo_predictor_keys)[[0, 5]])
    categorical_cols = np.array(demo_predictor_keys)[[1,2,3,4]]

    X, y = mm.create_data_input(all_predictors, start_time, IDcol, predictor_keys, demfeats_vals, labelsvals, numofhours)
    ##only with WFS >=3
    indices = (X.loc[:,'Mean_CONTINUOUS_MFS'] >= 3).index
    X  = X.loc[indices]
    y = y.loc[indices]
    ## train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state=2)
    #create validation set too
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2)

    ## create scaler and apply only to numeric data before adding binary data
    mapper = DataFrameMapper(
        [(continuous_col, StandardScaler()) for continuous_col in continuous_cols]
    )
    pipeline = Pipeline([("scaler", mapper), 
                         ("imputer", SimpleImputer(strategy="median")),
                        ('features', SelectKBest(k=k_feat))])
    X_train_norm = pipeline.fit_transform(X_train, y_train)

    return pipeline['features'].scores_, X_train_norm.shape[0]


# In[106]:


##load the data and create train/test split
def datacreator(client, path, numofhours):

    keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']
    target = 'label'
    IDcol = 'Shopid'
    hourscol = 'hours'
    start_time = 24*3
    k_feat = 70

    all_predictors, labelsvals, demfeats_vals = mm.load_data_all(univ_type=client,include_dems = True)
    predictor_keys = mm.get_predictor(keys)
    demo_predictor_keys = mm.get_dem_predictors() 

    continuous_cols = np.array(predictor_keys)
    continuous_cols = np.append(continuous_cols, np.array(demo_predictor_keys)[[0, 5]])
    categorical_cols = np.array(demo_predictor_keys)[[1,2,3,4]]

    X, y = mm.create_data_input(all_predictors, start_time, IDcol, predictor_keys, demfeats_vals, labelsvals, numofhours)
    ##only with WFS >=3
    indices = (X.loc[:,'Mean_CONTINUOUS_MFS'] >= 3).index
    X  = X.loc[indices]
    y = y.loc[indices]
    ## train-test split
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state=2)
    #create validation set too
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2)

    ## create scaler and apply only to numeric data before adding binary data
    mapper = DataFrameMapper(
        [(continuous_col, StandardScaler()) for continuous_col in continuous_cols]
    )

    pipeline = Pipeline([("scaler", mapper), 
                         ("imputer", SimpleImputer(strategy="median"))])
    X_train_norm = pipeline.fit_transform(X_train, y_train)
    features = np.loadtxt(f'{path}client/Data_subpopn/features_{numofhours}.csv')
    X_train_norm = pd.DataFrame(X_train_norm, index = X_train.index)
    X_train_norm = X_train_norm.iloc[:,features]
    #X_train_norm = X_train_norm.merge(X_train.iloc[:,-5:], left_index = True, right_index = True)
    
    ##apply scaler to test data
    X_test_norm = pipeline.transform(X_test)
    X_test_norm = pd.DataFrame(X_test_norm, index = X_test.index)
    X_test_norm = X_test_norm.iloc[:,features]
    #X_test_norm = X_test_norm.merge(X_test.iloc[:,-5:], left_index = True, right_index = True)

    ##apply scaler to val data
    #X_val_norm = pipeline.transform(X_val)
    #X_val_norm = pd.DataFrame(X_val_norm, index = X_val.index)
    #X_val_norm = X_val_norm.merge(X_val.iloc[:,-5:], left_index = True, right_index = True)
    X_val_norm = pd.DataFrame()
    y_val = pd.DataFrame()

    df_list = [X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val]
    writer = pd.ExcelWriter(f'{path}client/Data_subpopn/dataset_client_{client}_{numofhours}.xlsx')
    for i, df in enumerate(df_list):
        df.to_excel(writer,'sheet{}'.format(i))
        writer.save()   
    return 


# In[31]:


def dataloader(path, client):
    dfs = ['X_train_norm', 'X_test_norm', 'X_val_norm', 'y_train', 'y_test', 'y_val']
    for i, df in zip(range(6), dfs):
        globals()[df] = pd.read_excel(open(f'{path}client/Data_subpopn/dataset_client_{client}.xlsx', 'rb'),
                  sheet_name=f'sheet{i}', index_col = 0)
    return X_train_norm, X_test_norm, X_val_norm, y_train, y_test, y_val


# In[34]:


def main():
    # Read in the arguments provided by the master server
    parser = argparse.ArgumentParser()
    parser.add_argument('-cl','--client')
    parser.add_argument('-pt','--path', default = '/Users/ae2722/Documents/DCI_code/')
    parser.add_argument('-hr','--hours')
    parser.add_argument('-fs','--features')
    args = parser.parse_args()
    
    client = args.client
    path = args.path
    numofhours = int(args.hours)
    features = args.features
    # create the dataset
    if features == 'True':
        features, size = feature_selection(client, path, numofhours)
        print(size, features)
        
    else:
        datacreator(client, path, numofhours)

if __name__ == '__main__':
    main()

