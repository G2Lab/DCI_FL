
# import matplotlib.pyplot as plt
# import matplotlib.colors as colors
# import pickle

import csv
import itertools
import os
# import matplotlib.colors as colors
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy import stats
import scipy as sc
from scipy import stats
# from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import interp1d as interp
from scipy.stats import chi2
from sklearn import preprocessing
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
# from sklearn import  metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer

# from copy import deepcopy
from sklearn_pandas import DataFrameMapper
import helper.util_func as uf

# import scipy as sc
#
# import itertools
# import csv
# from copy import deepcopy
# from sklearn_pandas import DataFrameMapper
#
# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.pipeline import Pipeline
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import Imputer
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
#
# from sklearn import  metrics   #Additional scklearn functions
# from sklearn.model_selection import GridSearchCV, cross_val_score
# from sklearn.metrics import roc_curve, auc
# from sklearn.model_selection import StratifiedKFold
# # from scipy import interp
# from sklearn import svm
# from sklearn.metrics import confusion_matrix
# from sklearn import preprocessing
# import xgboost as xgb
# from sklearn.feature_selection import SelectKBest
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import AdaBoostClassifier
# 04/25/2022 - Implementing NN
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense,GRU
import tensorflow as tf

def percentile(n):
    def percentile_(x):
        return np.nanpercentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def IQR_Range(m,n):
    def IQR_Range_(x):
        return np.nanpercentile(x, n) - np.nanpercentile(x, m)
    IQR_Range_.__name__ = 'IQR_Range'
    return IQR_Range_

def maxminRange_(x):
    # non_zero_data = x[x!= 0]
    if np.isnan(np.nanmax(x)):
        return np.nan
    return np.nanmax(x)-np.nanmin(x)


def entropy_(x):
    # non_zero_data = x[x!= 0]
    non_zero_data = x[pd.notnull(x)]
    non_zero_data = non_zero_data[non_zero_data!=0]
    ep = sc.stats.entropy(non_zero_data)  # input probabilities to get the entropy
    if ep == -float('inf'):
        ep =0
    return ep



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    cm = np.around(cm, 2)
    plt.colorbar()
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cm


def load_csv(filename):
    b = []
    # print(filename);
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # print row
            b.append((row))
    return b

def run_other_models_NN(classifier,x_true,y_true,cv,name,continuous_cols,categorical_cols,labelsvals,predictor_keys,demo_predictor_keys,results_file_name,fillNa = False,toplot = False,k_feat = 33,compute_riskscore=False,dataval=None,include_dems=True):
    print(compute_riskscore)
    tprs = []
    aucs = []
    hl_all = []
    chi_all = []
    confusion_matrix_all = []
    mean_fpr = np.linspace(0, 1, 100)
    # mean_fpr = np.arange(0,1,100)
    probabvals_all_pos_all = []
    probabvals_all_neg_all = []
    i = 0
    coefvals = [];
    # shopids = labelsvals.keys()
    shopids = labelsvals.index.values
    if fillNa:
        mapper = DataFrameMapper(
            [(continuous_col, preprocessing.StandardScaler()) for continuous_col in continuous_cols]+
            [(categorical_col, preprocessing.LabelBinarizer()) for categorical_col in categorical_cols]
        )

    if toplot:
        plt.figure();
        plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    # temp = 1
    y_test_all = []
    y_probs_all = []
    mapper = DataFrameMapper(
        [(continuous_col, preprocessing.StandardScaler()) for continuous_col in continuous_cols] +
        [(categorical_col, preprocessing.LabelBinarizer()) for categorical_col in categorical_cols]
    )

    for train, test in cv.split(x_true, y_true): # outside cv
        X_train = x_true.iloc[train]
        y_train = y_true.iloc[train]

        data_preprocesisng_pipeline = Pipeline([('Normalize',mapper),
            ("imputer", Imputer(missing_values=np.nan,
                                                   strategy="median"))])

        data_preprocesisng_pipeline = data_preprocesisng_pipeline.fit(X_train)
        X_train = data_preprocesisng_pipeline.fit_transform(X_train)

        X_test = x_true.iloc[test]
        y_test = y_true.iloc[test]
        X_test = data_preprocesisng_pipeline.transform(X_test)

        shopid_test = shopids[test]
        model = get_NN_classifier(2,feature_size=X_train.shape[1])
        model.fit(X_train, y_train.to_numpy(), epochs=100,batch_size=100, validation_split=0.3,class_weight={0 : 0.4, 1 : 0.6})
        probas_ = model.predict(X_test)
    #
        if compute_riskscore:
            print(shopid_test.shape)
            print(x_true.shape)
            # (probabvals_all_pos, probabvals_all_neg) = compute_RiskScores(classifier,dataval,shopid_test,include_dems=include_dems)
            (probabvals_all_pos, probabvals_all_neg) = compute_RiskScores_NN(model, dataval, shopid_test, predictor_keys, demo_predictor_keys, name,
                               results_file_name, IDcol='Shopid', toplot=False, include_dems=include_dems,data_preprocesisng_pipeline=data_preprocesisng_pipeline)
            probabvals_all_pos_all.append(probabvals_all_pos)
            probabvals_all_neg_all.append(probabvals_all_neg)
            # print(probabvals_all_pos.shape)
            # print(probabvals_all_neg.shape)

        y_test_all.extend(y_test.values)
        y_probs_all.extend(probas_)

        hl_data = pd.DataFrame()
        hl_data['label'] = np.array(y_test.values).ravel()
        hl_data['prob'] = np.array(probas_)
        (df,hl,chi) = hl_test(hl_data, 10)
        hl_all.append(hl)
        chi_all.append(chi)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        # use optimal threshold instead of 0.5
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(optimal_threshold)
        y_scores2 = probas_ > optimal_threshold
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if toplot:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                      label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        cnf_matrix = confusion_matrix(y_test, y_scores2)
        confusion_matrix_all.append(cnf_matrix)
        i += 1
    if toplot:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    if toplot:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    if toplot:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.subplot2grid((3, 1), (2, 0), rowspan=1);
    # classifier_out = model.fit(data_preprocesisng_pipeline.fit_transform(x_true), y_true) # refit with all the data and save the final classifier, but we are not reporting AUCs on this
    # classifier_out =
    # model.fit(X_train, y_train.to_numpy(), epochs=100, batch_size=100, validation_split=0.3,
    #           class_weight={0: 0.4, 1: 0.6})
    cnf_mean_run = np.nanmedian(confusion_matrix_all, 0)
    print(cnf_mean_run)

    if toplot:
        plot_confusion_matrix(cnf_mean_run, classes={'DCI+', 'DCI-'}, normalize=True)
    return (confusion_matrix_all,(mean_auc, std_auc),model,np.array(coefvals),probabvals_all_pos_all,probabvals_all_neg_all,aucs,hl_all,chi_all)


def run_other_models(classifier,x_true,y_true,cv,name,continuous_cols,categorical_cols,labelsvals,predictor_keys,demo_predictor_keys,results_file_name,fillNa = False,toplot = False,k_feat = 33,compute_riskscore=False,dataval=None,include_dems=True):
    print(compute_riskscore)
    tprs = []
    aucs = []
    hl_all = []
    chi_all = []
    confusion_matrix_all = []
    mean_fpr = np.linspace(0, 1, 100)
    # mean_fpr = np.arange(0,1,100)
    probabvals_all_pos_all = []
    probabvals_all_neg_all = []
    i = 0
    coefvals = [];
    # shopids = labelsvals.keys()
    shopids = labelsvals.index.values
    if fillNa:

        mapper = DataFrameMapper(
            [(continuous_col, preprocessing.StandardScaler()) for continuous_col in continuous_cols]+
            [(categorical_col, preprocessing.LabelBinarizer()) for categorical_col in categorical_cols]
        )
        classifier = Pipeline([('Normalize',mapper),
            ("imputer", Imputer(missing_values=np.nan,
                                                   strategy="median")),
                               ("feature_selection",SelectKBest(k=k_feat)),
                               (name, classifier)])
    else:
        classifier = Pipeline([('Normalize',preprocessing.StandardScaler()),
                               (name, classifier)])

    if toplot:
        plt.figure();
        plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    # temp = 1
    y_test_all = []
    y_probs_all = []
    for train, test in cv.split(x_true, y_true): # outside cv
        X_train = x_true.iloc[train]
        y_train = y_true.iloc[train]
        X_test = x_true.iloc[test]
        y_test = y_true.iloc[test]
        shopid_test = shopids[test]

        classifier.fit(X_train, y_train) # classifier is Grid search CV with 5 fold -cv
        if name == 'LR' or name == 'RF' or name == 'EC':
            probas_ = classifier.predict_proba(X_test)
            y_scores2 = np.argmax(probas_, axis=1)
            probas_ = probas_[:, 1]
            if compute_riskscore:
                print(shopid_test.shape)
                print(x_true.shape)
                # (probabvals_all_pos, probabvals_all_neg) = compute_RiskScores(classifier,dataval,shopid_test,include_dems=include_dems)
                (probabvals_all_pos, probabvals_all_neg) = compute_RiskScores(classifier, dataval, shopid_test, predictor_keys, demo_predictor_keys, name,
                                   results_file_name, IDcol='Shopid', toplot=False, include_dems=include_dems)
                probabvals_all_pos_all.append(probabvals_all_pos)
                probabvals_all_neg_all.append(probabvals_all_neg)
                # print(probabvals_all_pos.shape)
                # print(probabvals_all_neg.shape)
        else:
            if hasattr(classifier, "predict_proba"):
                probas_ = classifier.predict_proba(X_test)
                classifier.score(X_test,y_test)
                # y_scores2 = np.argmax(probas_, axis=1)
                probas_ = probas_[:,1]
            else:  # use decision function
                prob_pos = classifier.decision_function(X_test)
                probas_ = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
                # y_scores2 = probas_ > 0.5
        y_test_all.extend(y_test.values)
        y_probs_all.extend(probas_)
        hl_data = pd.DataFrame()
        hl_data['label'] = np.array(y_test.values).ravel()
        hl_data['prob'] = np.array(probas_)
        (df,hl,chi) = hl_test(hl_data, 10)
        hl_all.append(hl)
        chi_all.append(chi)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        # use optimal threshold instead of 0.5
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(optimal_threshold)
        y_scores2 = probas_ > optimal_threshold
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        if toplot:
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                      label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        cnf_matrix = confusion_matrix(y_test, y_scores2)
        confusion_matrix_all.append(cnf_matrix)
        if name == 'SK' or name == 'EC':
            coefvals.append([])
        elif name == 'LR':
            coefvals.append(np.squeeze(classifier._final_estimator.coef_))
        elif name == 'RF' or name == 'AB':
            tempcoef = [i.base_estimator.best_estimator_.feature_importances_ for i in classifier._final_estimator.calibrated_classifiers_ ]
            tempcoef = np.nanmean(tempcoef,0)
            coefvals.append(np.squeeze(tempcoef))
        else:
            tempcoef = [i.base_estimator.best_estimator_.coef_ for i in classifier._final_estimator.calibrated_classifiers_ ]
            tempcoef = np.nanmean(tempcoef,0)
            coefvals.append(np.squeeze(tempcoef))
        i += 1
    if toplot:
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    # plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    if toplot:
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.subplot2grid((3, 1), (0, 0), rowspan=2);
    if toplot:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.subplot2grid((3, 1), (2, 0), rowspan=1);

    classifier_out = classifier.fit(x_true, y_true) # refit with all the data and save the final classifier, but we are not reporting AUCs on this
    cnf_mean_run = np.nanmedian(confusion_matrix_all, 0)
    print(cnf_mean_run)

    # hl_data = pd.DataFrame()
    # hl_data['label'] = np.array(y_test_all).ravel()
    # hl_data['prob'] = np.array(y_probs_all)
    # hl_test(hl_data,10)

    # y_hat = classifier_out.predict_proba(x_true)
    # hl_data['label'] = np.array(y_true.values).ravel()
    # hl_data['prob'] = y_hat[:,1]
    # hl_test(hl_data,10)
    # y_scores2 = np.array(y_probs_all) > 0.45
    # cnf_matrix = confusion_matrix(np.array(y_test_all).ravel(), y_scores2)
    # fpr, tpr, thresholds = roc_curve(np.array(y_test_all).ravel(),  np.array(y_probs_all))
    # roc_auc = auc(fpr, tpr)

    # fraction_of_positives, mean_predicted_value = calibration_curve(hl_data['label'], hl_data['prob'], n_bins=8)
    # plt.figure(figsize=(10, 10))
    # ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    # ax2 = plt.subplot2grid((3, 1), (2, 0))
    # ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    # ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
    #          label="%s" % (name, ))
    #
    # ax2.hist(hl_data['prob'], range=(0, 1), bins=8, label=name,
    #          histtype="step", lw=2)
    # ax1.set_ylabel("Fraction of positives")
    # ax1.set_ylim([-0.05, 1.05])
    # ax1.legend(loc="lower right")
    # ax1.set_title('Calibration plots  (reliability curve)')
    #
    # ax2.set_xlabel("Mean predicted value")
    # ax2.set_ylabel("Count")
    # ax2.legend(loc="upper center", ncol=2)
    #
    # plt.tight_layout()
    # plt.show()
    # plt.scatter(np.arange(1,102),hl_data.loc[hl_data['label']==1]['prob'])
    # plt.boxplot([hl_data.loc[hl_data['label'] == 0]['prob'], hl_data.loc[hl_data['label'] == 1]['prob']])
    # # plt.hlines(0.5,0,311)
    # plt.scatter(np.arange(1+102, 102+210), hl_data.loc[hl_data['label'] == 0]['prob'])
    if toplot:
        plot_confusion_matrix(cnf_mean_run, classes={'DCI+', 'DCI-'}, normalize=True)
    return (confusion_matrix_all,(mean_auc, std_auc),classifier_out,np.array(coefvals),probabvals_all_pos_all,probabvals_all_neg_all,aucs,hl_all,chi_all)


def entropy_1(data1):
    non_zero_data = data1[data1 != 0]
    entropy1 = sc.stats.entropy(non_zero_data)  # input probabilities to get the entropy
    return entropy1

def compute_RiskScores_NN(classifer_eval,dataval,valid_shop_id,predictor_keys,demo_predictor_keys,name,results_file_name ,IDcol='Shopid',toplot=True,include_dems=True,data_preprocesisng_pipeline=None):

    datatouse = 1;  # 1 or 24 hours
    if toplot:
        plt.figure();
    probabvals_all_pos = ();
    probabvals_all_neg = ();
    # for shopidvals in ShopId_test:
    # for shopidvals in np.arange(1,352,1):
    for shopidvals in valid_shop_id:
        # np.arange(10,352,150):
        # print(shopidvals)
        datapat = dataval[dataval.Shopid == shopidvals]
        # datapat = datapat.ffill();  # forward fill the missing values
        timevals = ()
        probabvals = ()
        # for numofhours in np.arange(12, 168+12, 12):
        for numofhours in np.arange(12, 168 + 12, 1):
            numberofhourslookback = numofhours;
            df_filtered = datapat[(datapat.hours >= 0) & (
                    datapat.hours <=  numberofhourslookback )]
            df_pat_hour = datapat[(datapat.hours >= numberofhourslookback-12) & (
                    datapat.hours <=  numberofhourslookback )].aggregate(
                    [np.nanmedian])
            datatoprocess = df_filtered;
            ## code for adding demographics variable
            if include_dems:
                demfeats_vals = datatoprocess[[IDcol] + demo_predictor_keys].groupby(IDcol).aggregate(
                    [np.nanmedian])  # get median and standard deviation
            # featurevals = datatoprocess.groupby(IDcol).aggregate(
            #     [np.nanmean, np.nanstd])  # get median and standard deviation
            featurevals = datatoprocess[[IDcol] + predictor_keys].groupby(IDcol).agg(
                [maxminRange_, np.nanmean, np.nanstd, np.nanmedian, IQR_Range(25, 75),
                 entropy_])  # get median and standard deviation
            # labelsvals = featurevals['labels']['nanmedian']
            labelsvals = np.nanmean(df_filtered.label)
            if include_dems:
                featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
            # end code for adding demographics data
            featurevals = featurevals.dropna(axis=0, thresh=35)
            timevals = np.append(timevals, numberofhourslookback)
            if featurevals.shape[0] > 0:
                # labelsvals = featurevals['labels']['nanmedian']
                # featurevals = featurevals[predictors]
                featurevals = data_preprocesisng_pipeline.transform(featurevals)
                probabval_hour = classifer_eval.predict(featurevals)
                if len(df_pat_hour) == 0:
                    probabval_hour = np.nan
                if np.sum(np.isnan(df_pat_hour[predictor_keys].T).values) > 0.6 * len(predictor_keys) :
                    probabval_hour = np.nan
                probabvals = np.append(probabvals, probabval_hour)
            else:
                probabvals = np.append(probabvals, np.nan)
        if labelsvals == 1:
            # plt.plot(timevals/24,probabvals,color='r',linewidth = 2.5)
            if len(probabvals_all_pos) == 0:
                probabvals_all_pos = probabvals
            else:
                probabvals_all_pos = np.vstack([probabvals_all_pos, probabvals])
        else:
            # plt.plot(timevals/24, probabvals, color='g',linewidth = 2.5)
            if len(probabvals_all_neg) == 0:
                probabvals_all_neg = probabvals
            else:
                probabvals_all_neg = np.vstack([probabvals_all_neg, probabvals])

    if toplot:
        df_filtered = df_filtered.reset_index()
        tempdf = pd.DataFrame({'probs':probabvals})
        tempdf = pd.concat(([tempdf,df_filtered[predictor_keys]]),ignore_index=False,axis=1)

        # df_filtered.iloc[:,0:10].plot(subplots=True)
        tempdf.iloc[:, 0:20].plot(subplots=True)

        plt.figure()
        plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_neg, axis=0), linewidth=2.5, color='g')
        plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0), linewidth=2.5, color='r')
        plt.fill_between(timevals / 24 -7, np.nanmedian(probabvals_all_neg, axis=0) - np.nanstd(probabvals_all_neg, axis=0),
                         np.nanmedian(probabvals_all_neg, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
                         edgecolor='g', facecolor='g')
        plt.fill_between(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0) - np.nanstd(probabvals_all_pos, axis=0),
                         np.nanmedian(probabvals_all_pos, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
                         edgecolor='r', facecolor='r')

        plt.xlabel('Time(days from DCI)', fontsize=13, fontweight='bold')
        plt.ylabel('Risk Score', fontsize=13, fontweight='bold')
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
        plt.yticks(fontsize=13, fontweight='bold')
        plt.xticks((-6, -5, -4, -3, -2, -1, 0), (-6, -5, -4, -3, -2, -1, 'DCI'), fontsize=13, fontweight='bold',
                   rotation=45)
        plt.legend(('DCI-', 'DCI+'), loc=2, prop=dict(size=13, weight='bold'))
        plt.tight_layout()
        plt.savefig( results_file_name + name+'_Time_series.png')
        plt.close('all')
    return (probabvals_all_pos,probabvals_all_neg)

def compute_RiskScores(classifer_eval,dataval,valid_shop_id,predictor_keys,demo_predictor_keys,name,results_file_name ,IDcol='Shopid',toplot=True,include_dems=True):

    datatouse = 1;  # 1 or 24 hours
    if toplot:
        plt.figure();
    probabvals_all_pos = ();
    probabvals_all_neg = ();
    # for shopidvals in ShopId_test:
    # for shopidvals in np.arange(1,352,1):
    for shopidvals in valid_shop_id:
        # np.arange(10,352,150):
        # print(shopidvals)
        datapat = dataval[dataval.Shopid == shopidvals]
        # datapat = datapat.ffill();  # forward fill the missing values
        timevals = ()
        probabvals = ()
        # for numofhours in np.arange(12, 168+12, 12):
        for numofhours in np.arange(12, 168 + 12, 1):
            numberofhourslookback = numofhours;
            df_filtered = datapat[(datapat.hours >= 0) & (
                    datapat.hours <=  numberofhourslookback )]
            # datapat.iloc[:,8:23].plot(subplots=True)
            # df_filtered.iloc[:,8:23].plot(subplots=True)
            # df_filtered.shape
            # code to check if more than 50 % of predictorkeys are NAN at that hour, if yes then do not compute probab
            df_pat_hour = datapat[(datapat.hours >= numberofhourslookback-12) & (
                    datapat.hours <=  numberofhourslookback )].aggregate(
                    [np.nanmedian])

            datatoprocess = df_filtered;
            ## code for adding demographics variable
            if include_dems:
                demfeats_vals = datatoprocess[[IDcol] + demo_predictor_keys].groupby(IDcol).aggregate(
                    [np.nanmedian])  # get median and standard deviation
            # featurevals = datatoprocess.groupby(IDcol).aggregate(
            #     [np.nanmean, np.nanstd])  # get median and standard deviation
            featurevals = datatoprocess[[IDcol] + predictor_keys].groupby(IDcol).agg(
                [maxminRange_, np.nanmean, np.nanstd, np.nanmedian, IQR_Range(25, 75),
                 entropy_])  # get median and standard deviation
            # labelsvals = featurevals['labels']['nanmedian']
            labelsvals = np.nanmean(df_filtered.label)
            if include_dems:
                featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
            # end code for adding demographics data
            featurevals = featurevals.dropna(axis=0, thresh=35)
            timevals = np.append(timevals, numberofhourslookback)
            if featurevals.shape[0] > 0:
                # labelsvals = featurevals['labels']['nanmedian']
                # featurevals = featurevals[predictors]
                if name == 'FALSE':
                    probabval_hour = classifer_eval.decision_function(featurevals)
                else:
                    probabval_hour = classifer_eval.predict_proba(featurevals)[:, 1]

                if len(df_pat_hour) == 0:
                    probabval_hour = np.nan
                if np.sum(np.isnan(df_pat_hour[predictor_keys].T).values) > 0.6 * len(predictor_keys)  :
                    probabval_hour = np.nan

                probabvals = np.append(probabvals, probabval_hour)
            else:
                probabvals = np.append(probabvals, np.nan)
        if labelsvals == 1:
            # plt.plot(timevals/24,probabvals,color='r',linewidth = 2.5)
            if len(probabvals_all_pos) == 0:
                probabvals_all_pos = probabvals
            else:
                probabvals_all_pos = np.vstack([probabvals_all_pos, probabvals])
        else:
            # plt.plot(timevals/24, probabvals, color='g',linewidth = 2.5)
            if len(probabvals_all_neg) == 0:
                probabvals_all_neg = probabvals
            else:
                probabvals_all_neg = np.vstack([probabvals_all_neg, probabvals])

    if toplot:
        df_filtered = df_filtered.reset_index()
        tempdf = pd.DataFrame({'probs':probabvals})
        tempdf = pd.concat(([tempdf,df_filtered[predictor_keys]]),ignore_index=False,axis=1)

        # df_filtered.iloc[:,0:10].plot(subplots=True)
        tempdf.iloc[:, 0:20].plot(subplots=True)

        plt.figure()
        plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_neg, axis=0), linewidth=2.5, color='g')
        plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0), linewidth=2.5, color='r')
        plt.fill_between(timevals / 24 -7, np.nanmedian(probabvals_all_neg, axis=0) - np.nanstd(probabvals_all_neg, axis=0),
                         np.nanmedian(probabvals_all_neg, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
                         edgecolor='g', facecolor='g')
        plt.fill_between(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0) - np.nanstd(probabvals_all_pos, axis=0),
                         np.nanmedian(probabvals_all_pos, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
                         edgecolor='r', facecolor='r')

        plt.xlabel('Time(days from DCI)', fontsize=13, fontweight='bold')
        plt.ylabel('Risk Score', fontsize=13, fontweight='bold')
        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
        plt.yticks(fontsize=13, fontweight='bold')
        plt.xticks((-6, -5, -4, -3, -2, -1, 0), (-6, -5, -4, -3, -2, -1, 'DCI'), fontsize=13, fontweight='bold',
                   rotation=45)
        plt.legend(('DCI-', 'DCI+'), loc=2, prop=dict(size=13, weight='bold'))
        plt.tight_layout()
        plt.savefig( results_file_name + name+'_Time_series.png')
        plt.close('all')
    return (probabvals_all_pos,probabvals_all_neg)



def entropy_1(data1):
    non_zero_data = data1[data1 != 0]
    entropy1 = sc.stats.entropy(non_zero_data)  # input probabilities to get the entropy
    return entropy1


def norm_conf_matrices(cnf_all):
    norm_cnf_all =[]
    for cm in cnf_all:
        norm_cnf_all.append(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    return norm_cnf_all
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     # K.get_session().run(tf.local_variables_initializer())
#     return auc

def get_NN_classifier(cv, feature_size=138, hidden_layers = 2, hidden_layer_size = 200):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=feature_size, activation='relu'))
    model.add(tf.keras.layers.Masking(mask_value=np.nan))
    for hl in np.arange(0,hidden_layers-1,1):
        model.add(Dense(hidden_layer_size,activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
    return model

##TODO: LSTM/GRU LAYER
def get_LSTM_classifier(feature_size=138, hidden_layers = 2, hidden_layer_size = 200):
    model = Sequential()
    model.add(Dense(hidden_layer_size, input_dim=feature_size, activation='relu'))
    model.add(tf.keras.layers.Masking(mask_value=np.nan))
    for hl in np.arange(0,hidden_layers-1,1):
        model.add(GRU(hidden_layer_size,activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.AUC()])
    return model

def get_classifiers_list(cv):
    classifier_list = []
    randomstate = 7
    # Define Logistic Regression Classifier
    Cs = np.logspace(-10, 10, 10)
    lr = LogisticRegressionCV(Cs=Cs, penalty='l2', class_weight='balanced', scoring='roc_auc',solver='liblinear',
                                      refit=True, cv=cv, random_state=randomstate,n_jobs=-1, dual=False,max_iter=5000)


    classifier_list.append([lr,'LR'])

    # Define calibrated SVM - classifier
    Cs = np.logspace(-5, 5, 10)
    svc = svm.LinearSVC(penalty='l1', class_weight='balanced', random_state=randomstate,dual=False, max_iter=5000)
    svc = GridSearchCV(estimator=svc, param_grid=dict(C=Cs),  scoring='roc_auc', refit=True, cv=cv,n_jobs=-1)
    svc = CalibratedClassifierCV(svc)
    classifier_list.append([svc, 'SL'])

    # Define Calibrated SVM-K Classifier
    svck = svm.SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=randomstate)
    gammas = np.logspace(-6, 6, 25)
    svck = GridSearchCV(estimator=svck, param_grid=dict(C=Cs, gamma=gammas),  scoring='roc_auc', refit=True,
                        cv=cv,n_jobs=-1)
    svck = CalibratedClassifierCV(svck,cv=cv)
    classifier_list.append([svck, 'SK'])


    #Define RF classifier
    rfc = RandomForestClassifier(n_estimators=5,random_state=randomstate,class_weight='balanced')
    param_grid = {
        'n_estimators': [5, 10, 20],
        'max_features': ['auto','log2'],
        "bootstrap": [True, False],
        'criterion': ["gini", "entropy"]
    }
    # rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, refit=True, scoring='roc_auc', cv=5)
    n_iter_search = 16
    rfc = RandomizedSearchCV(rfc, param_distributions=param_grid,
                                       n_iter=n_iter_search, cv=cv,n_jobs=-1,scoring='roc_auc')
    rfc=CalibratedClassifierCV(rfc,cv=cv)
    classifier_list.append([rfc, 'RF'])

    # eclf1 = VotingClassifier(estimators=[('lr', lr),('svck',svck),('rfc',rfc),
    #                                      ('SVC', svm.SVC(kernel='linear', probability=True,class_weight='balanced'))],voting='soft')
    eclf1 = VotingClassifier(estimators=[('lr', lr),('svck',svck),('rfc',rfc),
                                         ('SVC', svc)],voting='soft')
    eclf2 = VotingClassifier(estimators=[('lr', lr),('svck',svck),
                                         ('SVC', svc)],voting='soft')
    # eclf1 = VotingClassifier(estimators=[('lr', lr),('svck',svck),
    #                                      ('SVC', CalibratedClassifierCV(svc))],voting='soft')
    classifier_list.append([eclf1, 'EC'])

    return classifier_list

# Define Hosmer Lemeshow for goodness of fit

def hl_test(data, g):
    '''
    Hosmer-Lemeshow test to judge the goodness of fit for binary data

    Input: dataframe(data), integer(num of subgroups divided)

    Output: float
    '''
    data_st = data.sort_values('prob')
    data_st['dcl'] = pd.qcut(data_st['prob'], g,duplicates='drop')

    ys = data_st['label'].groupby(data_st.dcl).sum()
    yt = data_st['label'].groupby(data_st.dcl).count()
    yn = yt - ys

    yps = data_st['prob'].groupby(data_st.dcl).sum()
    ypt = data_st['prob'].groupby(data_st.dcl).count()
    ypn = ypt - yps

    hltest = (((ys - yps) ** 2 / yps) + ((yn - ypn) ** 2 / ypn)).sum()
    pval = 1 - chi2.cdf(hltest, g - 2)

    df = g - 2

    print('\n HL-chi2({}): {}, p-value: {}\n'.format(df, hltest, pval))
    return (df,hltest,pval)

def get_predictor(keys):
    predictor_keys = []
    for key in keys:
        for func, name in zip([np.nanmean, np.nanstd], ['Mean', 'STD']):
            predictor_keys.append(name+'_'+key)
            if key != 'HR' and name == 'Mean':
                predictor_keys.append(name+'_max_Xcorr_HR_'+key)
                predictor_keys.append(name +'_min_Xcorr_HR_' + key)
    return predictor_keys

def get_dem_predictors():
    dem_keys = ['Mean_CONTINUOUS_Age', 'Mean_CATEGORICAL_Sex', 'Mean_CONTINUOUS_HH', 'Mean_CONTINUOUS_MFS', 'Mean_CONTINUOUS_WFNS', 'Mean_CONTINUOUS_GCS']
    return dem_keys

def get_predictor_data(anchordir,all_predictor_keys,dci_index = 'Mean_DCI_Index'):
    all_predictors = pd.DataFrame()
    for i, file_name in enumerate(sorted(os.listdir(anchordir))):
        # if '2007' in file_name:
            print(file_name)
            print(i)
            if not os.path.isfile(os.path.join(anchordir,file_name)):
                continue
            # pid = file_name.split('_ | .')[1]
            pid = re.split('_|, |.pickle', file_name)[1]
            # break
            fin = open(os.path.join(anchordir,file_name), 'rb')
            patient = pickle.load(fin, encoding='latin1')
            fin.close()
            print(patient.keys())
            # np.sum(patient['CONTINUOUS_WFNS'])

            ## add predictor vals add nan values if the key is missing
            a  = [apk not in patient.keys().values for apk in all_predictor_keys]
            missing_keys = np.array(all_predictor_keys)[a]
            a = [patient.insert(2,apk,np.nan)for apk in missing_keys]

            pat_predictor_vals = patient[all_predictor_keys]


            ## add shop ids vals
            pat_predictor_vals['Shopid'] = [int(pid)]  * pat_predictor_vals.shape[0]

            #add labels
            pat_predictor_vals['label'] = patient[dci_index]

            # add hours
            pat_predictor_vals['hours'] = np.arange(1, patient.shape[0] + 1, 1)

            #append all the predictors in one feature matrix
            all_predictors = pd.concat([all_predictors,pat_predictor_vals],ignore_index=True)
    return all_predictors



## get max Classifier before DCI and plot the AUCs over time
def get_max_classifier_before_dci(result_hours,classifier_name='EC',classifier_at_dci=False,classifier_at_hour=-3.5,toplot=False):
    for color, name in [
        # ('b', 'EC'),
        # ('g', 'SL'),
        # ('r', 'LR'),
        # ('c', 'SK'),
        # ('m', 'RF')
        ('darkcyan', 'EC'),
        ('darkcyan', 'SL'),
        ('darkcyan', 'LR'),
        ('darkcyan', 'SK'),
        ('darkcyan', 'RF')

    ]:
        if name==classifier_name:
            print(name)
            mean_auc_vals = ()
            time_vals = ()
            std_auc_vals = ()
            classifier_all = ()
            for p_id, p_info in result_hours.items():
                if name in p_id:
                    print("\n", p_info['N'], '\t', p_info['N_positive'], '\t', p_id, '\t AUC \t', p_info['mean_auc'],
                          '\t +/- \t', p_info['std_auc'])
                    mean_auc_vals = np.append(mean_auc_vals, np.median(p_info['AUCs_all']))
                    std_auc_vals = np.append(std_auc_vals, p_info['std_auc'])
                    time_vals = np.append(time_vals, p_info['Hours'])
                    classifier_all = np.append(classifier_all, p_info['Classifier'])

            ## sort it based on the time
            sortidx = np.unravel_index(np.argsort(time_vals, axis=None), time_vals.shape)
            sortetimeval = time_vals[sortidx]/24 - 7
            sorted_classifier_all = classifier_all[sortidx]
            sorted_mean_auc = mean_auc_vals[sortidx] # sorted based on time not AUC values
            if toplot:
                plt.figure(1)
                plt.errorbar(sortetimeval,mean_auc_vals[sortidx],std_auc_vals[sortidx],linewidth = 2.5,color=color,elinewidth=0.5,label=name, fmt = '')

            #get classifier before 0
            if classifier_at_dci:
                idx_before_dci = sortetimeval==0#-3.5#6.5 #TODO: Change it to 0
            else:
                idx_before_dci = sortetimeval==classifier_at_hour
                # idx_before_dci = sortetimeval<0
            sortetimeval = sortetimeval[idx_before_dci]
            sorted_mean_auc = sorted_mean_auc[idx_before_dci]
            sorted_classifier_all = sorted_classifier_all[idx_before_dci]
            print('Max at ',sortetimeval[np.argmax(sorted_mean_auc)],' hrs before anchor with AUC: ',np.max(sorted_mean_auc))
            return (sorted_classifier_all[np.argmax(sorted_mean_auc)],np.max(sorted_mean_auc))
    return []



def compute_RiskScores_Impala(classifer_eval,datapat,name,keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR'],IDcol='Shopid',include_dems=True,compute_hours = 1):
        predictor_keys =get_predictor(keys)
        demo_predictor_keys = get_dem_predictors()
        # datapat = pat_predictor_vals
        timevals = ()
        timevals1=()
        probabvals = ()
        # for numofhours in np.arange(12, 168+12, 12):
        for numofhours in np.arange(0, len(datapat), compute_hours):
            numberofhourslookback = numofhours;
            df_filtered = datapat[(datapat.hours >= 0) & (
                    datapat.hours <=  numberofhourslookback )]
            if df_filtered.empty:
                continue
            # datapat.iloc[:,8:23].plot(subplots=True)
            # df_filtered.iloc[:,8:23].plot(subplots=True)
            # df_filtered.shape
            # code to check if more than 50 % of predictorkeys are NAN at that hour, if yes then do not compute probab
            df_pat_hour = datapat[(datapat.hours >= numberofhourslookback-12) & (
                    datapat.hours <=  numberofhourslookback )].aggregate(
                    [np.nanmedian])

            datatoprocess = df_filtered;
            ## code for adding demographics variable
            if include_dems:
                demfeats_vals = df_filtered[[IDcol] + demo_predictor_keys].groupby(IDcol).aggregate(
                    [np.nanmedian])  # get median and standard deviation
            # featurevals = datatoprocess.groupby(IDcol).aggregate(
            #     [np.nanmean, np.nanstd])  # get median and standard deviation
            featurevals = datatoprocess[[IDcol] + predictor_keys].groupby(IDcol).agg(
                [maxminRange_, np.nanmean, np.nanstd, np.nanmedian, IQR_Range(25, 75),
                 entropy_])  # get median and standard deviation
            # labelsvals = featurevals['labels']['nanmedian']
            # labelsvals = np.nanmean(df_filtered.label)
            if include_dems:
                featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
            # end code for adding demographics data
            featurevals = featurevals.dropna(axis=0, thresh=35)
            timevals = np.append(timevals, numberofhourslookback)
            timevals1 = np.append(timevals1, datatoprocess.iloc[-1]['_timesstamp'])
            if featurevals.shape[0] > 0:
                # labelsvals = featurevals['labels']['nanmedian']
                # featurevals = featurevals[predictors]
                if name == 'FALSE':
                    probabval_hour = classifer_eval.decision_function(featurevals)
                else:
                    probabval_hour = classifer_eval.predict_proba(featurevals)[:, 1]

                if len(df_pat_hour) == 0:
                    probabval_hour = np.nan
                if np.sum(np.isnan(df_pat_hour[predictor_keys].T).values) > 0.6 * len(predictor_keys)  :
                    probabval_hour = np.nan

                probabvals = np.append(probabvals, probabval_hour)
            else:
                probabvals = np.append(probabvals, np.nan)
        return (probabvals,timevals1)
    #     if labelsvals == 1:
    #         # plt.plot(timevals/24,probabvals,color='r',linewidth = 2.5)
    #         if len(probabvals_all_pos) == 0:
    #             probabvals_all_pos = probabvals
    #         else:
    #             probabvals_all_pos = np.vstack([probabvals_all_pos, probabvals])
    #     else:
    #         # plt.plot(timevals/24, probabvals, color='g',linewidth = 2.5)
    #         if len(probabvals_all_neg) == 0:
    #             probabvals_all_neg = probabvals
    #         else:
    #             probabvals_all_neg = np.vstack([probabvals_all_neg, probabvals])
    #
    # if toplot:
    #     df_filtered = df_filtered.reset_index()
    #     tempdf = pd.DataFrame({'probs':probabvals})
    #     tempdf = pd.concat(([tempdf,df_filtered[predictor_keys]]),ignore_index=False,axis=1)
    #
    #     # df_filtered.iloc[:,0:10].plot(subplots=True)
    #     tempdf.iloc[:, 0:20].plot(subplots=True)
    #
    #     plt.figure()
    #     plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_neg, axis=0), linewidth=2.5, color='g')
    #     plt.plot(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0), linewidth=2.5, color='r')
    #     plt.fill_between(timevals / 24 -7, np.nanmedian(probabvals_all_neg, axis=0) - np.nanstd(probabvals_all_neg, axis=0),
    #                      np.nanmedian(probabvals_all_neg, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
    #                      edgecolor='g', facecolor='g')
    #     plt.fill_between(timevals / 24 - 7, np.nanmedian(probabvals_all_pos, axis=0) - np.nanstd(probabvals_all_pos, axis=0),
    #                      np.nanmedian(probabvals_all_pos, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
    #                      edgecolor='r', facecolor='r')
    #
    #     plt.xlabel('Time(days from DCI)', fontsize=13, fontweight='bold')
    #     plt.ylabel('Risk Score', fontsize=13, fontweight='bold')
    #     plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
    #     plt.yticks(fontsize=13, fontweight='bold')
    #     plt.xticks((-6, -5, -4, -3, -2, -1, 0), (-6, -5, -4, -3, -2, -1, 'DCI'), fontsize=13, fontweight='bold',
    #                rotation=45)
    #     plt.legend(('DCI-', 'DCI+'), loc=2, prop=dict(size=13, weight='bold'))
    #     plt.tight_layout()
    #     plt.savefig( results_file_name + name+'_Time_series.png')
    #     plt.close('all')
    # return (probabvals_all_pos,probabvals_all_neg)

## Computes Xcorr and features for the DCI Model
def compute_xcorr_feats_Impala(philips1_resample_wide,dem_feats_value,keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']):

    philips1_resample_wide = philips1_resample_wide.rename(columns={"ARTm": "ART-M", "ARTs": "ART-S","ARTd": "ART-D","SpO₂":"SPO2"})

    ##Step 3 # Compute X_corr
    patient = uf.compute_cross_corr(philips1_resample_wide,keys=keys)

    ## step 4 - Compute Features
    #  4.1 Consolidate hourly values - in two steps
    #
    # take hourly average
    # a. taking 10 mins mean,SD, Median,
    philips1_resample_10 = patient.resample('10min', on='timestamp').agg(['mean','median','std']).reset_index()

    # b taking 60 mins (6 values of 10 mins ) mean,SD, Median, so make sure that the length is divisible by 10 if not append nans to the end and then take average
    philips1_resample_60 = philips1_resample_10.resample('60min', on='timestamp').agg('mean').reset_index()
    # drop the level and rename the columns
    philips1_resample_60.columns = [x[1].upper()+'_'+x[0] if 'std' in x[1] else x[1].capitalize()+'_'+x[0] if isinstance(x, tuple) else x for x in philips1_resample_60.columns.ravel()]

    #  4.2 - Compute Featuress and generate risk scores
    predictor_keys = get_predictor(keys)
    demo_predictor_keys = get_dem_predictors()
    all_predictor_keys = predictor_keys + demo_predictor_keys
    a = [apk not in philips1_resample_60.keys().values for apk in all_predictor_keys]
    missing_keys = np.array(all_predictor_keys)[a]  ## Demographic variabiles are missing at this point

    a = [philips1_resample_60.insert(2, apk, np.nan) for apk in missing_keys] ## Adding NaNs (to all the missing data, as the ML pipeline has intepolration logic)

    ## Add demographic Features
    # results = [philips1_resample_60[x] for x,y in zip(dem_feats_value['Features'],dem_feats_value['Value'])]
    philips1_resample_60[dem_feats_value['Features']]=dem_feats_value['Value']

    # philips1_resample_60[demo_predictor_keys].head().T = 0 ## TODO Replace with actual demographic values
    pat_predictor_vals = philips1_resample_60[all_predictor_keys]


    pat_predictor_vals['hours'] = np.arange(1, pat_predictor_vals.shape[0] + 1, 1)
    pat_predictor_vals['Shopid'] = [1] * pat_predictor_vals.shape[0]
    pat_predictor_vals['_timesstamp'] = philips1_resample_60['_timestamp']
    return pat_predictor_vals