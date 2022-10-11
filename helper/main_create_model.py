import helper.util_ml as uml
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import _pickle as pickle
from sklearn.model_selection import StratifiedKFold
from copy import deepcopy
import csv
import warnings
warnings.filterwarnings("ignore")

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()

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

def get_predictor_data(anchordir,all_predictor_keys):
    all_predictors = pd.DataFrame()
    for file_name in sorted(os.listdir(anchordir)):
        # if '2007' in file_name:
        if not os.path.isdir(os.path.join(anchordir, file_name)):
            #print(file_name)
            pid = file_name.split('_')[1]
            # break
            fin = open(os.path.join(anchordir,file_name), 'rb')
            patient = pickle.load(fin, encoding='latin1')
            fin.close()
            ## add predictor vals
            pat_predictor_vals = patient[all_predictor_keys]

            ## add shop ids vals
            pat_predictor_vals['Shopid'] = [pid]  * pat_predictor_vals.shape[0]

            #add labels
            pat_predictor_vals['label'] = patient['Mean_DCI_Index']

            # add hours
            pat_predictor_vals['hours'] = np.arange(1, patient.shape[0] + 1, 1)

            #append all the predictors in one feature matrix
            all_predictors = pd.concat([all_predictors,pat_predictor_vals],ignore_index=True)
    return all_predictors

def train_models_NN(all_predictors,demfeats_vals,labelsvals,continuous_cols,categorical_cols,all_classifiers,results_file_name,k_feat=60,include_dems=True,going_back_from_dci = False, start_time=24*3):
    results_hours = dict()
    anchortimepoint = 168 # since we have 14*24 = 336 hours data, anchor will be at 336/2 = 168
    # for numofhours in np.arange(12, 336+12, 12):  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
    if going_back_from_dci == True:
        range_idx = np.flip(np.arange(0, 168, 12))
    else:
        range_idx = np.arange(start_time, 168+12, 12) # TODO: WE ARE CREATING MODELS GOING FORWARD STARTING FROM DAY=3,24*3 hrs
    for numofhours in range_idx:  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
        print(numofhours)
        if going_back_from_dci:
            df_filtered = all_predictors[(all_predictors.hours <= 168) & (
                all_predictors.hours > numofhours)]
            # np.unique(df_filtered.hours)
        else:
            df_filtered = all_predictors[(all_predictors.hours >= start_time) & (
                all_predictors.hours <= numofhours)]

        featurevals = df_filtered[[IDcol] + predictor_keys].groupby(IDcol).agg(
            [uml.maxminRange_, np.nanmean, np.nanstd, np.nanmedian, uml.IQR_Range(25, 75),
             uml.entropy_])
        all_predictors.keys().T
        all_predictors.shape
        if include_dems:
            featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
            featurevals = featurevals.dropna(axis=0, thresh=35)
        else:
            featurevals = featurevals.dropna(axis=0, thresh=35)

        labelsvals_hours = labelsvals[labelsvals.index.isin(featurevals.index)]

        featurevals = featurevals.astype(float)

        for classifier_ ,name in zip([nn_classifier],['NN']):
            print(name)
            compute_riskscore = True
            if numofhours >=48 :
                compute_riskscore = True

            (cnf_mean_run, (mean_auc, std_auc), classifier_, coefvals, probabvals_all_pos_all,
             probabvals_all_neg_all, auc_all,hl_all,chi_all) = uml.run_other_models_NN(classifier_,
                                                                 x_true=featurevals,
                                                                 y_true=labelsvals_hours, cv=cv, name =name,
                                                                 continuous_cols= continuous_cols,
                                                                 categorical_cols=categorical_cols,labelsvals=labelsvals_hours,predictor_keys = predictor_keys, demo_predictor_keys=demo_predictor_keys,results_file_name=results_file_name,
                                                                 fillNa=True, k_feat=k_feat, toplot=False,#k_feat=60
                                                                 compute_riskscore=compute_riskscore, dataval=all_predictors,
                                                                 include_dems=include_dems)


            if numofhours >=48 and compute_riskscore:
                a=dict([])
                a['pos']  = probabvals_all_pos_all
                a['neg'] = probabvals_all_neg_all
                pickle.dump(a, open(results_file_name + "riskscores_"+name +"_"+str(numofhours)+ "__hourly.p", "wb"))


            print(str(np.round(np.median(auc_all), 3)) + ' ' + str(
                np.round(np.percentile(auc_all, [25, 75]), 3)))

                # print(classifier_)

            results_predictor = dict();
            results_predictor['Median_Confusion_Matrix'] = np.nanmedian(cnf_mean_run, 0)
            results_predictor['Mean_Confusion_Matrix'] = np.nanmean(cnf_mean_run, 0)
            results_predictor['std_Confusion_Matrix'] = np.nanstd(cnf_mean_run, 0)
            results_predictor['mean_auc'] = mean_auc
            results_predictor['std_auc'] = std_auc
            results_predictor['N'] = featurevals.shape[0]
            results_predictor['N_positive'] = np.sum(labelsvals.values)
            results_predictor['Hours'] = (numofhours)
            results_predictor['Classifier'] = deepcopy(classifier_)
            results_predictor['Coefs'] = (coefvals)
            results_predictor['AUCs_all'] = (auc_all)
            results_predictor['CM_all'] = (cnf_mean_run)
            results_predictor['HL_all'] = (hl_all)
            results_predictor['CHI_all'] = (chi_all)
            results_hours['All_Feats' + str(numofhours) + '_' + name] = results_predictor
    return  results_hours

def train_models(all_predictors,demfeats_vals,labelsvals,continuous_cols,categorical_cols,all_classifiers,results_file_name, predictor_keys, cv, demo_predictor_keys,k_feat=60,include_dems=True,going_back_from_dci = False, start_time=24*3):
    target = 'label'
    IDcol = 'Shopid'
    hourscol = 'hours'

    results_hours = dict()
    anchortimepoint = 168 # since we have 14*24 = 336 hours data, anchor will be at 336/2 = 168
    # for numofhours in np.arange(12, 336+12, 12):  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
    if going_back_from_dci == True:
        range_idx = np.flip(np.arange(0, 168, 12))
    else:
        range_idx = np.arange(start_time, 168+12, 12) # TODO: WE ARE CREATING MODELS GOING FORWARD STARTING FROM DAY=3,24*3 hrs
    for numofhours in range_idx:  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
        print(numofhours)
        # if numofhours !=168:
        #     continue
        if going_back_from_dci:
            df_filtered = all_predictors[(all_predictors.hours  <=  168) & (
                all_predictors.hours   > numofhours)]
            # np.unique(df_filtered.hours)
        else:
            df_filtered = all_predictors[(all_predictors.hours >= start_time) & (
                all_predictors.hours   <= numofhours)]

        featurevals = df_filtered[[IDcol] + predictor_keys].groupby(IDcol).agg(
            [uml.maxminRange_, np.nanmean, np.nanstd, np.nanmedian, uml.IQR_Range(25, 75),
             uml.entropy_])
        all_predictors.keys().T
        all_predictors.shape
        if include_dems:
            featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
            featurevals = featurevals.dropna(axis=0, thresh=35)
        else:
            featurevals = featurevals.dropna(axis=0, thresh=35)

        labelsvals_hours = labelsvals[labelsvals.index.isin(featurevals.index)]

        featurevals = featurevals.astype(float)

        for classifier_ ,name in all_classifiers:
            if not (name == 'EC' ):
                continue
            print(name)
            compute_riskscore = True
            if numofhours >=48 :
                compute_riskscore = True

            (cnf_mean_run, (mean_auc, std_auc), classifier_, coefvals, probabvals_all_pos_all,
             probabvals_all_neg_all, auc_all,hl_all,chi_all) = uml.run_other_models(classifier_,
                                                                 x_true=featurevals,
                                                                 y_true=labelsvals_hours, cv=cv, name =name,
                                                                 continuous_cols= continuous_cols,
                                                                 categorical_cols=categorical_cols,labelsvals=labelsvals_hours,predictor_keys = predictor_keys, demo_predictor_keys=demo_predictor_keys,results_file_name=results_file_name,
                                                                 fillNa=True, k_feat=k_feat, toplot=False,#k_feat=60
                                                                 compute_riskscore=compute_riskscore, dataval=all_predictors,
                                                                 include_dems=include_dems)


            if numofhours >=48 and compute_riskscore:
                a=dict([])
                a['pos']  = probabvals_all_pos_all
                a['neg'] = probabvals_all_neg_all
                pickle.dump(a, open(results_file_name + "riskscores_"+name +"_"+str(numofhours)+ "__hourly.p", "wb"))


            print(str(np.round(np.median(auc_all), 3)) + ' ' + str(
                np.round(np.percentile(auc_all, [25, 75]), 3)))

                # print(classifier_)

            results_predictor = dict();
            results_predictor['Median_Confusion_Matrix'] = np.nanmedian(cnf_mean_run, 0)
            results_predictor['Mean_Confusion_Matrix'] = np.nanmean(cnf_mean_run, 0)
            results_predictor['std_Confusion_Matrix'] = np.nanstd(cnf_mean_run, 0)
            results_predictor['mean_auc'] = mean_auc
            results_predictor['std_auc'] = std_auc
            results_predictor['N'] = featurevals.shape[0]
            results_predictor['N_positive'] = np.sum(labelsvals.values)
            results_predictor['Hours'] = (numofhours)
            results_predictor['Classifier'] = deepcopy(classifier_)
            results_predictor['Coefs'] = (coefvals)
            results_predictor['AUCs_all'] = (auc_all)
            results_predictor['CM_all'] = (cnf_mean_run)
            results_predictor['HL_all'] = (hl_all)
            results_predictor['CHI_all'] = (chi_all)
            results_hours['All_Feats' + str(numofhours) + '_' + name] = results_predictor
    return  results_hours

def write_results_csv(results_hours,results_file_name,going_back_from_dci=True, start_time=24*3,name='NN'):
    w = csv.writer(open(results_file_name, "w"))
    w.writerow(
        [' ', 'LR', ' ', ' ', ' ', ' ', ' ', 'SL', ' ', ' ', ' ', ' ', ' ', 'SK', ' ', ' ', ' ', ' ', ' ', 'RF', ' ',
         ' ', ' ', ' ', ' ', 'EC', ' ', ' ', ' ', ' ', ' '])
    w.writerow(
        ['Time from DCI', 'AUC', 'AUC_med', 'TP', 'FP', 'FN', 'TN', 'AUC', 'AUC_med', 'TP', 'FP', 'FN', 'TN', 'AUC',
         'AUC_med', 'TP', 'FP', 'FN', 'TN', 'AUC', 'AUC_med', 'TP', 'FP', 'FN', 'TN', 'AUC', 'AUC_med', 'TP', 'FP',
         'FN', 'TN'])
    # for numofhours in np.arange(-144, 180, 12):
    # for numofhours in np.arange(12, 336 + 12, 12):  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
    if going_back_from_dci == True:
        range_idx = np.flip(np.arange(0, 168, 12))
    else:
        # range_idx = np.arange(12, 168+12, 12)
        range_idx = np.arange(start_time, 168 + 12,
                              12)  # TODO: WE ARE CREATING MODELS GOING FORWARD STARTING FROM DAY=3,24*3 hrs

    for numofhours in range_idx:  # -156,156,24 - that ways -12 to 12 will be the full day with DCI
        print(numofhours)
        results_val = list()
        results_val.append(str(numofhours / 24))
        # for name in ['LR', 'SL', 'SK', 'RF', 'EC']:
        for name in [name]:
            print(name)
            result_predictor = results_hours['All_Feats' + str(numofhours) + '_' + name]
            cnf_all = result_predictor['CM_all']
            norm_cnf_all = uml.norm_conf_matrices(cnf_all)
            cm25, cm, cm75 = np.percentile(norm_cnf_all, [25, 50, 75], axis=0)
            AUCs = str(np.round(result_predictor['mean_auc'], 2)) + ' +/- ' + str(
                np.round(result_predictor['std_auc'], 3))
            AUCs_M = str(np.round(np.median(result_predictor['AUCs_all']), 2)) + ' ' + str(
                np.round(np.percentile(result_predictor['AUCs_all'], [25, 75]), 2))
            std_cm = result_predictor['std_Confusion_Matrix'] / cm.sum(axis=1)[:, np.newaxis]
            TP = str(np.round(cm[0, 0], 2)) + ' [' + str(np.round(cm25[0, 0], 2)) + '-' + str(
                np.round(cm75[0, 0], 2)) + ']'
            FP = str(np.round(cm[0, 1], 2)) + ' [' + str(np.round(cm25[0, 1], 2)) + '-' + str(
                np.round(cm75[0, 1], 2)) + ']'
            FN = str(np.round(cm[1, 0], 2)) + ' [' + str(np.round(cm25[1, 0], 2)) + '-' + str(
                np.round(cm75[1, 0], 2)) + ']'
            TN = str(np.round(cm[1, 1], 2)) + ' [' + str(np.round(cm25[1, 1], 2)) + '-' + str(
                np.round(cm75[1, 1], 2)) + ']'
            results_val.append(AUCs)
            results_val.append(AUCs_M)
            results_val.append(TP)
            results_val.append(FP)
            results_val.append(FN)
            results_val.append(TN)
        print(results_val)
        w.writerow(results_val)
def xtick_label_vals_feats(include_dems=True):

    xticklabelsvals = [

        # # ICP
        'ICP Mean R',
        'ICP Mean M',
        'ICP Mean S',
        'ICP Mean Med',
        'ICP Mean IQRR',
        'ICP Mean E',

        'ICP SD R',
        'ICP SD M',
        'ICP SD S',
        'ICP SD Med',
        'ICP SD IQRR',
        'ICP SD E',

        'Xcorr HR-ICP R',
        'Xcorr HR-ICP M',
        'Xcorr HR-ICP S',
        'Xcorr HR-ICP Med',
        'Xcorr HR-ICP IQRR',
        'Xcorr HR-ICP E',


        # RR
        'RR Mean R',
        'RR Mean M',
        'RR Mean S',
        'RR Mean Med',
        'RR Mean IQRR',
        'RR Mean E',

        'RR SD R',
        'RR SD M',
        'RR SD S',
        'RR SD Med',
        'RR SD IQRR',
        'RR SD E',

        'Xcorr HR-RR R',
        'Xcorr HR-RR M',
        'Xcorr HR-RR S',
        'Xcorr HR-RR Med',
        'Xcorr HR-RR IQRR',
        'Xcorr HR-RR E',

        # 'SPO2
        'SPO2 Mean R',
        'SPO2 Mean M',
        'SPO2 Mean S',
        'SPO2 Mean Med',
        'SPO2 Mean IQRR',
        'SPO2 Mean E',

        'SPO2 SD R',
        'SPO2 SD M',
        'SPO2 SD S',
        'SPO2 SD Med',
        'SPO2 SD IQRR',
        'SPO2 SD E',

        'Xcorr HR-SPO2 R',
        'Xcorr HR-SPO2 M',
        'Xcorr HR-SPO2 S',
        'Xcorr HR-SPO2 Med',
        'Xcorr HR-SPO2 IQRR',
        'Xcorr HR-SPO2 E',

        # ARD
        'ARD Mean R',
        'ARD Mean M',
        'ARD Mean S',
        'ARD Mean Med',
        'ARD Mean IQRR',
        'ARD Mean E',

        'ARD SD R',
        'ARD SD M',
        'ARD SD S',
        'ARD SD Med',
        'ARD SD IQRR',
        'ARD SD E',

        'Xcorr HR-ARD R',
        'Xcorr HR-ARD M',
        'Xcorr HR-ARD S',
        'Xcorr HR-ARD Med',
        'Xcorr HR-ARD IQRR',
        'Xcorr HR-ARD E',

        # ARS
        'ARS Mean R',
        'ARS Mean M',
        'ARS Mean S',
        'ARS Mean Med',
        'ARS Mean IQRR',
        'ARS Mean E',

        'ARS SD R',
        'ARS SD M',
        'ARS SD S',
        'ARS SD Med',
        'ARS SD IQRR',
        'ARS SD E',

        'Xcorr HR-ARS R',
        'Xcorr HR-ARS M',
        'Xcorr HR-ARS S',
        'Xcorr HR-ARS Med',
        'Xcorr HR-ARS IQRR',
        'Xcorr HR-ARS E',

        # # # TEMP
        'TEMP Mean R',
        'TEMP Mean M',
        'TEMP Mean S',
        'TEMP Mean Med',
        'TEMP Mean IQRR',
        'TEMP Mean E',

        'TEMP SD R',
        'TEMP SD M',
        'TEMP SD S',
        'TEMP SD Med',
        'TEMP SD IQRR',
        'TEMP SD E',

        'Xcorr HR-TEMP R',
        'Xcorr HR-TEMP M',
        'Xcorr HR-TEMP S',
        'Xcorr HR-TEMP Med',
        'Xcorr HR-TEMP IQRR',
        'Xcorr HR-TEMP E',


        'HR Mean R',
        'HR Mean M',
        'HR Mean S',
        'HR Mean Med',
        'HR Mean IQRR',
        'HR Mean E',

        'HR SD R',
        'HR SD M',
        'HR SD S',
        'HR SD Med',
        'HR SD IQRR',
        'HR SD E'

    ]
    if include_dems:
        xticklabelsvals.extend(['Age',
        'Sex',
        'HH_Adm',
        'MFS',
        'WFNS',
        'GCS'])
    return  xticklabelsvals

def plot_results(results_hours,xticklabelsvals,all_predictors,file_name):
    for color, name in [
        ('b', 'EC'),
        ('g', 'SL'),
        ('r', 'LR'),
        ('c', 'SK'),
        ('m', 'RF')
    ]:
        print(name)
        mean_auc_vals = ()
        std_auc_vals = ()
        time_vals = ()
        classifier_all = ()
        coefval_time = ()
        coefval_time2 = ()
        counter = 1
        for p_id, p_info in results_hours.items():
            if name in p_id:
                print("\n", p_info['N'], '\t', p_info['N_positive'], '\t', p_id, '\t AUC \t', p_info['mean_auc'],
                      '\t +/- \t', p_info['std_auc'])
                p_info['HL_all']
                p_info['CHI_all']
                np.round(np.median(p_info['AUCs_all']), 3)
                mean_auc_vals = np.append(mean_auc_vals, np.median(p_info['AUCs_all']))
                std_auc_vals = np.append(std_auc_vals, p_info['std_auc'])
                time_vals = np.append(time_vals, p_info['Hours'])
                classifier_all = np.append(classifier_all, p_info['Classifier'])
                b = np.where(p_info['Classifier'].get_params()['feature_selection'].get_support())
                xticklabelsvals1 = np.array(xticklabelsvals)
                # print(xticklabelsvals1[b])
                if counter == 1:
                    b = np.where(p_info['Classifier'].get_params()[
                                     'feature_selection'].get_support())  # since only k best are choosen we will have to assign those to correct locations
                    coefval_time = np.nan * np.zeros(250)  ## TODO: Get this dynamically
                    if p_info['Coefs'].shape[1] > 0:
                        coefval_time[b] = np.mean(p_info['Coefs'], axis=0)
                    counter = counter + 1;
                else:
                    if np.mean(p_info['Coefs'], axis=0).shape[0] == 0:
                        continue
                    b = np.where(p_info['Classifier'].get_params()[
                                     'feature_selection'].get_support())  # since only k best are choosen we will have to assign those to correct locations
                    coefval_temp = np.nan * np.zeros(250) ## TODO: Get this dynamically
                    if p_info['Coefs'].shape[1] > 0:
                        coefval_temp[b] = np.mean(p_info['Coefs'], axis=0)
                    coefval_time = np.vstack([coefval_time, coefval_temp]);
                    # coefval_time2 = np.vstack([coefval_time2, np.squeeze(p_info['Classifier']._final_estimator.coef_)]);

        sortidx = np.unravel_index(np.argsort(time_vals, axis=None), time_vals.shape)
        sortetimeval = time_vals[sortidx] / 24 -7
        plt.errorbar(sortetimeval, mean_auc_vals[sortidx], std_auc_vals[sortidx], linewidth=2.5, color=color,
                     elinewidth=0.5, label=name, fmt='')

        plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
        plt.axvline(x=0, color='k', linestyle='-', linewidth=2.5)

        sortetimeval = np.asarray(sortetimeval, dtype='str')
        sortetimeval[sortetimeval == '0.0'] = 'DCI'
        plt.yticks(fontsize=12, fontweight='bold');
        plt.xticks(time_vals[sortidx]/24-7,sortetimeval,fontsize=13,fontweight='bold',rotation=45)
        # plt.xticks(np.arange(-7, 8, 1), np.arange(-7, 8, 1), fontsize=13, fontweight='bold', rotation=45)
        plt.ylim([0.4, 1.00])
        # plt.xlim([-6.5, 7.5])
        plt.xlabel('Time(days)', fontsize=13, fontweight='bold')
        plt.ylabel('AUCs', fontsize=13, fontweight='bold')
        plt.text(0.51, 0.51, 'Chance level', fontsize=13, fontweight='bold')
        plt.legend(prop={'size': 10, 'weight': 'bold'})
        # plt.show()
        plt.tight_layout()
        plt.savefig(file_name+name+'.png')
        plt.savefig(file_name + name + '.pdf')
        plt.close()
        # if coefval_time.shape[1] != 0:
        if len(coefval_time.shape) > 1:
            coefval_time = coefval_time[np.squeeze(np.asarray(sortidx)), :]  # sort it based on the time index
            plt.figure(figsize=(7.30, 7.69))
            bounds = np.array([-1, -0.5, -0.125, 0, 0.125, 0.5, 1])
            norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
            norm = colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                                     vmin=-1.0, vmax=1.0)
            plt.imshow(nanzscore(np.transpose(coefval_time)), interpolation="none",
                       extent=[-6, 7, coefval_time.shape[1], 0], cmap=plt.get_cmap('PiYG'),
                       # norm=norm
                       vmin=-3.5, vmax=3.5
                       );
            plt.colorbar()
            ax = plt.gca()
            ax.set_aspect(0.25)
            plt.xlabel('Time(days)', fontsize=13, fontweight='bold')
            plt.ylabel('Feature (weights)', fontsize=13, fontweight='bold')
            plt.axvline(x=0, color='k', linestyle='-', linewidth=2.5)

            plt.yticks(np.arange(0, coefval_time.shape[1], 1), xticklabelsvals, fontsize=6, rotation=15)

            plt.axvline(x=0, color='k', linestyle='-', linewidth=2.5)
            # plt.show()
            plt.tight_layout()
            plt.savefig(file_name+'_feats_'+name+'.png')
            plt.savefig(file_name + '_feats_' + name + '.pdf')
            plt.close()

def plot_risk_scores(risk_score_file_name,name,vmin=.25,vmax=.50):
    fin = open(risk_score_file_name, 'rb')
    a = pickle.load(fin, encoding='latin1')
    fin.close()

    timevals = np.arange(12, 168 + 12, 12) - 168
    timevals = np.arange(-168, 0, 1)

    pos_all = a['pos']
    neg_all = a['neg']

    if len(pos_all) == 0:
        return

    len(pos_all)
    for i in np.arange(0,5):
        print(len(pos_all[i]))

    probabvals_all_pos = np.concatenate(np.array(pos_all))
    probabvals_all_neg = np.concatenate(np.array(neg_all))
    med_rs = np.nanmedian(probabvals_all_pos, axis=0)
    q75, q25 = np.nanpercentile(probabvals_all_pos, [75, 25], axis=0)
    print('Risk scores median (IQR) pos : ', np.around(med_rs[-1], 2), '(', np.around(q25[-1], 2), '-',
          np.around(q75[-1], 2), ')')

    med_rs = np.nanmedian(probabvals_all_neg, axis=0)
    q75, q25 = np.nanpercentile(probabvals_all_neg, [75, 25], axis=0)
    print('Risk scores median (IQR) pos : ', np.around(med_rs[-1], 2), '(', np.around(q25[-1], 2), '-',
          np.around(q75[-1], 2), ')')
    mean_plot = True

    probabvals_all = np.concatenate([probabvals_all_neg,probabvals_all_pos],axis=0)

    plt.figure()
    plt.imshow((probabvals_all), cmap=plt.get_cmap('coolwarm'), zorder=1,vmin=vmin,vmax=vmax)
    ax = plt.gca()
    ax.set_aspect(0.5)
    plt.axhline(y=len(probabvals_all_neg), color='k', linestyle='-', linewidth=2.5)
    plt.colorbar()
    plt.tight_layout()
    plt.title(risk_score_file_name,fontsize=13, fontweight='bold')
    plt.xticks(np.arange(0,len(timevals),2),timevals[0::2])



    plt.figure()
    if mean_plot:
        plt.plot(timevals / 24, np.nanmean(probabvals_all_neg, axis=0), linewidth=2.5, color='g')
        plt.plot(timevals / 24, np.nanmean(probabvals_all_pos, axis=0), linewidth=2.5, color='r')

        plt.fill_between(timevals / 24, np.nanmean(probabvals_all_neg, axis=0) - np.nanstd(probabvals_all_neg, axis=0),
                         np.nanmean(probabvals_all_neg, axis=0) + np.nanstd(probabvals_all_neg, axis=0), alpha=0.05,
                         edgecolor='g', facecolor='g')
        plt.fill_between(timevals / 24, np.nanmean(probabvals_all_pos, axis=0) - np.nanstd(probabvals_all_pos, axis=0),
                         np.nanmean(probabvals_all_pos, axis=0) + np.nanstd(probabvals_all_pos, axis=0), alpha=0.05,
                         edgecolor='r', facecolor='r')
        plt.plot(timevals / 24, np.nanmean(probabvals_all_neg, axis=0) - np.nanstd(probabvals_all_neg, axis=0),
                 color='g', alpha=0.1)
        plt.plot(timevals / 24, np.nanmean(probabvals_all_neg, axis=0) + np.nanstd(probabvals_all_neg, axis=0),
                 color='g', alpha=0.1)
        plt.plot(timevals / 24, np.nanmean(probabvals_all_pos, axis=0) - np.nanstd(probabvals_all_pos, axis=0),
                 color='r', alpha=0.1)
        plt.plot(timevals / 24, np.nanmean(probabvals_all_pos, axis=0) + np.nanstd(probabvals_all_pos, axis=0),
                 color='r', alpha=0.1)
    else:
        plt.plot(timevals / 24, np.nanmedian(probabvals_all_neg, axis=0), linewidth=2.5, color='g')
        plt.plot(timevals / 24, np.nanmedian(probabvals_all_pos, axis=0), linewidth=2.5, color='r')
        plt.fill_between(timevals / 24, np.nanpercentile(probabvals_all_neg, [25], axis=0)[0],
                         np.nanpercentile(probabvals_all_neg, [75], axis=0)[0], alpha=0.05,
                         edgecolor='g', facecolor='g')
        plt.fill_between(timevals / 24, np.nanpercentile(probabvals_all_pos, [25], axis=0)[0],
                         np.nanpercentile(probabvals_all_pos, [75], axis=0)[0], alpha=0.05,
                         edgecolor='r', facecolor='r')
        plt.plot(timevals / 24, np.nanpercentile(probabvals_all_neg, [25], axis=0)[0], color='g', alpha=0.1)
        plt.plot(timevals / 24, np.nanpercentile(probabvals_all_neg, [75], axis=0)[0], color='g', alpha=0.1)
        plt.plot(timevals / 24, np.nanpercentile(probabvals_all_pos, [75], axis=0)[0], color='r', alpha=0.1)
        plt.plot(timevals / 24, np.nanpercentile(probabvals_all_pos, [25], axis=0)[0], color='r', alpha=0.1)

    plt.xlabel('Time(days from Anchor)', fontsize=13, fontweight='bold')
    plt.ylabel('Risk Score', fontsize=13, fontweight='bold')
    plt.axhline(y=0.35, color='k', linestyle='--', linewidth=1.5)
    # plt.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5)
    plt.yticks(fontsize=13, fontweight='bold')
    plt.xticks((-6, -5, -4, -3, -2, -1, 0), (-6, -5, -4, -3, -2, -1, 'Anchor'), fontsize=13, fontweight='bold',
               rotation=45)
    plt.title(risk_score_file_name,fontsize=13, fontweight='bold')
    plt.legend(('DCI-', 'DCI+'), loc=2, prop=dict(size=13, weight='bold'))
    plt.tight_layout()

def plot_tpv_fpv_over_time(risk_score_file_name):
    fin = open(risk_score_file_name, 'rb')
    a = pickle.load(fin, encoding='latin1')
    fin.close()
    timevals = np.arange(12, 168 + 12, 12) - 168

    pos_all = a['pos']
    neg_all = a['neg']
    # pos_all = a[0]
    # neg_all = a[1]
    threshold_val = 0.35
    len(pos_all)
    for i in np.arange(0, 5):
        print(len(pos_all[i]))

    probabvals_all_pos = np.concatenate(np.array(pos_all))
    probabvals_all_neg = np.concatenate(np.array(neg_all))
    med_rs = np.nanmedian(probabvals_all_pos, axis=0)
    q75, q25 = np.nanpercentile(probabvals_all_pos, [75, 25], axis=0)
    print('Risk scores median (IQR) pos : ', np.around(med_rs[-1], 2), '(', np.around(q25[-1], 2), '-',
          np.around(q75[-1], 2), ')')
    threshold_val = np.around(med_rs[-1], 2)
    med_rs = np.nanmedian(probabvals_all_neg, axis=0)
    q75, q25 = np.nanpercentile(probabvals_all_neg, [75, 25], axis=0)
    print('Risk scores median (IQR) pos : ', np.around(med_rs[-1], 2), '(', np.around(q25[-1], 2), '-',
          np.around(q75[-1], 2), ')')
    mean_plot = True
    threshold_val = threshold_val + np.around(med_rs[-1], 2)
    threshold_val = threshold_val / 2

    propb_neg_pd = pd.DataFrame(probabvals_all_neg)
    propb_pos_pd = pd.DataFrame(probabvals_all_pos)
    temp = [propb_neg_pd < threshold_val]
    discharge_correctly = [np.sum(temp[0][i]) for i in temp[0]]
    temp = [propb_pos_pd < threshold_val]
    discharge_incorrectly = [np.sum(temp[0][i]) for i in temp[0]]

    npv_over_time = [TN / (TN + FN) for (TN, FN) in zip(discharge_correctly, discharge_incorrectly)]
    specificity = [number / propb_neg_pd.shape[0] for number in discharge_correctly]

    temp = [propb_neg_pd > threshold_val]
    dci_incorrectly = [np.sum(temp[0][i]) for i in temp[0]]
    temp = [propb_pos_pd > threshold_val]
    dci_correctly = [np.sum(temp[0][i]) for i in temp[0]]

    ppv_over_time = [TP / (TP + FP) for (TP, FP) in zip(dci_correctly, dci_incorrectly)]
    sensitivity = [number / propb_pos_pd.shape[0] for number in dci_correctly]

    plt.figure(1)
    plt.subplot(3, 2, 1)
    # plt.bar(np.arange(0,len(npv_over_time),1)+w,npv_over_time,width=0.1,label=risk_score_file_name)
    plt.plot(npv_over_time, label=risk_score_file_name)
    plt.title('NPV = TN/(TN+FN)')


    # plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(specificity, label=risk_score_file_name)
    plt.title('Specificity - Discharge Correctly (TN)/Total number of negative patients')

    plt.subplot(3, 2, 3)
    plt.plot(ppv_over_time, label=risk_score_file_name)
    plt.title('PPV = TP/(TP+FP)')

    plt.subplot(3, 2, 4)
    plt.plot(sensitivity, label=risk_score_file_name)
    plt.title('sensitivity - DCI Predicted(TP)/Total number of DCI patients')

    plt.subplot(3, 2, 6)
    plt.plot(1, label=risk_score_file_name)
    plt.legend()
    # plt.tight_layout()

def load_data_all(univ_type='all',include_dems = True):
    target = 'label'
    IDcol = 'Shopid'
    hourscol = 'hours'
    ## load all the pickles and append it in one array

    keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']
    predictor_keys = get_predictor(keys)
    demo_predictor_keys = get_dem_predictors()
    if include_dems:
        # get all the predictor keys
        all_predictor_keys = predictor_keys + demo_predictor_keys
    else:
        all_predictor_keys = predictor_keys
    all_predictors_univs_combined = pd.DataFrame([])
    labelsvals_univs_combined = pd.DataFrame([])
    demfeats_vals_univs_combined = pd.DataFrame([])
    if univ_type == 'all':
        var_list = ['CUMC','Aachen','UTH']
    else:
        var_list = [univ_type]

    for univ in var_list : # create separate models for each university, and also with the combined data
        anchordir = '//prometheus.neuro.columbia.edu//neurocriticalcare//data//Projects//33_Federated_Learning_data//anchor//' + univ + '//'
        # ***********************2 . Get all the predictors*******************
        all_predictors = get_predictor_data(anchordir, all_predictor_keys)
        all_predictors = all_predictors.astype(float)
        temp = all_predictors.groupby(['Shopid']).ffill()  # forward fill the missing values
        all_predictors[temp.keys()] = temp

        # NO NEED TO AGGR WE CAN TAKE UNIQUE TOO
        labelsvals = all_predictors[[IDcol, target]].groupby(IDcol).aggregate(
            np.nanmedian)  # get median and standard deviation
        shopid = all_predictors[[IDcol, target]].groupby(IDcol).groups.keys()
        if include_dems:
            demfeats_vals = all_predictors[[IDcol] + demo_predictor_keys].groupby(IDcol).aggregate(
                [np.nanmedian])  # get median and standard deviation
        else:
            demfeats_vals = []

        all_predictors_univs_combined = pd.concat([all_predictors_univs_combined, all_predictors])
        labelsvals_univs_combined = pd.concat([labelsvals_univs_combined, labelsvals])
        demfeats_vals_univs_combined = pd.concat([demfeats_vals_univs_combined, demfeats_vals])

    return all_predictors_univs_combined,labelsvals_univs_combined,demfeats_vals_univs_combined

def create_data_input(all_predictors, start_time, IDcol, predictor_keys, demfeats_vals, labelsvals, numofhours):
    anchortimepoint = 168   
    results_hours = dict()
    df_filtered = df_filtered = all_predictors[(all_predictors.hours <= 168) & (
                all_predictors.hours > numofhours)]
    featurevals = df_filtered[[IDcol] + predictor_keys].groupby(IDcol).agg(
        [uml.maxminRange_, np.nanmean, np.nanstd, np.nanmedian, uml.IQR_Range(25, 75),
         uml.entropy_])
    all_predictors.keys().T
    all_predictors.shape
    featurevals = pd.concat([featurevals, demfeats_vals], join='inner', axis=1)
    featurevals = featurevals.dropna(axis=0, thresh=35)
    labelsvals_hours = labelsvals[labelsvals.index.isin(featurevals.index)]
    featurevals = featurevals.astype(float)
    return featurevals, labelsvals_hours


if __name__ == '__main__':
    #********************1 . Define Parameters***********************
    nanzscore = lambda x: (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0)
    randomstate = 7 # for reproducing results
    include_dems = True
    # anchordir = '/media/sf_Murad/Pickles_HR_SPO2/anchor/CUMC/'
    for univ in ['CUMC','Aachen','UTH','all']: # create separate models for each university, and also with the combined data
        # anchordir = '/mnt/H/Murad/Pickles_HR_SPO2/anchor/CUMC/'
        # if not univ in ['all']:
        #     continue
        # if univ in ['all']:
        #     load_data_all()
        anchordir = '/mnt/H/Murad/Pickles_HR_SPO2/anchor/'+univ+'/'
        keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']
        target = 'label'
        IDcol = 'Shopid'
        hourscol = 'hours'

        all_predictors, labelsvals, demfeats_vals = load_data_all(univ_type='all')
        ## load all the pickles and append it in one array

        predictor_keys = get_predictor(keys)
        demo_predictor_keys = get_dem_predictors()
        # if include_dems:
        #     # get all the predictor keys
        #     all_predictor_keys = predictor_keys + demo_predictor_keys
        # else:
        #     all_predictor_keys = predictor_keys
        #
        #
        # # ***********************2 . Get all the predictors*******************
        # all_predictors = get_predictor_data(anchordir,all_predictor_keys)
        # all_predictors = all_predictors.astype(float)
        # temp = all_predictors.groupby(['Shopid']).ffill()  # forward fill the missing values
        # all_predictors[temp.keys()]=temp
        #
        # #NO NEED TO AGGR WE CAN TAKE UNIQUE TOO
        # labelsvals = all_predictors[[IDcol, target]].groupby(IDcol).aggregate(np.nanmedian)  # get median and standard deviation
        #
        # # shopid = all_predictors[[IDcol, target]].groupby(IDcol).groups.keys()
        #
        # if include_dems:
        #     demfeats_vals = all_predictors[[IDcol] + demo_predictor_keys].groupby(IDcol).aggregate(
        #         [np.nanmedian])  # get median and standard deviation
        # else:
        #     demfeats_vals = []
        #
        # ## identify continous columns vs categorical columns for normalization during ML algorithms running
        continuous_cols = np.array(predictor_keys)

        if include_dems:
            continuous_cols = np.append(continuous_cols, np.array(demo_predictor_keys)[[0, 5]])
            categorical_cols = np.array(demo_predictor_keys)[[1,2,3,4]]
        else:
            categorical_cols = []

        # **********************3. Get all the classifiers********************
        cv = StratifiedKFold(n_splits=5)
        if univ == 'UTH':
            cv = StratifiedKFold(n_splits=3) # few patients, so 5 fold nested will fail

        going_back_from_dci = True
        if going_back_from_dci:
            if include_dems:
                result_dir = "results_NO_ICP_TEMP_strt_dci/"+univ+"/"
            else:
                result_dir = "results_NO_ICP_TEMP_strt_dci_no_dem/"+univ+"/"
        else:
            if include_dems:
                result_dir = "results_NO_ICP_TEMP/"+univ+"/"
            else:
                result_dir = "results_NO_ICP_TEMP_no_dem/"+univ+"/"
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)

        k_feat=70
        start_time = 24*4 # starting form 3 days before DCI for going forward instead of the first time point as we have more data since PBD 3
        for k_feat in [70]:#np.arange(20,80,10):
            print(k_feat)
            if not os.path.isdir(result_dir+str(k_feat)):
                os.mkdir(result_dir+str(k_feat))
            if not os.path.isdir(result_dir + str(k_feat)+"/figures"):
                os.mkdir(result_dir + str(k_feat)+"/figures")
            ## *****************************4. TRAIN CLASSIFIERS **************************************
            all_classifiers = uml.get_classifiers_list(cv)
            results_file_name = result_dir+str(k_feat)+"/"
            results_hours = train_models(all_predictors, demfeats_vals, labelsvals, continuous_cols, categorical_cols, all_classifiers,results_file_name=results_file_name,k_feat=k_feat,
                         include_dems=include_dems,going_back_from_dci=going_back_from_dci,start_time=start_time)

            ## *********************5. SAVE RESULTS*********************
            pickle.dump(results_hours, open(result_dir+str(k_feat)+"/ML_Results_cumulative_one_min_RR_"+str(k_feat)+".p", "wb"))
            ## ************************* write Results *****************
            write_results_csv(results_hours, result_dir+str(k_feat)+'/ML_Results_cumulative_one_min_RR_'+str(k_feat)+'.csv',going_back_from_dci=going_back_from_dci,start_time=start_time,name = 'EC')


        ##TODO: Create Deep learning based classifer
        nn_classifier = uml.get_NN_classifier(cv)
        results_file_name = result_dir + 'NN' + "/"
        if not os.path.isdir(result_dir + 'NN'):
            os.mkdir(result_dir + 'NN')
        if not os.path.isdir(result_dir + 'NN' + "/figures"):
            os.mkdir(result_dir + 'NN'+ "/figures")

        results_hours = train_models_NN(all_predictors, demfeats_vals, labelsvals, continuous_cols, categorical_cols,
                                     None, results_file_name=results_file_name, k_feat=k_feat,
                                     include_dems=include_dems, going_back_from_dci=going_back_from_dci,
                                     start_time=start_time)

        ## *********************5. SAVE RESULTS*********************
        pickle.dump(results_hours,
                    open(result_dir + 'NN' + "/ML_Results_cumulative_one_min_RR_" + str(k_feat) + ".p", "wb"))
        ## ************************* write Results *****************
        write_results_csv(results_hours,
                          result_dir + 'NN' + '/ML_Results_cumulative_one_min_RR_' + str(k_feat) + '.csv',
                          going_back_from_dci=going_back_from_dci, start_time=start_time,)


        #**********************6. Plot and Save Figures*****************
        # for k_feat in np.arange(70, 80, 10):
        for k_feat in np.arange(20, 80, 10):
            xticklabelsvals = xtick_label_vals_feats(include_dems=True)
            fin = open(
                result_dir + str(k_feat) + "/ML_Results_cumulative_one_min_RR_" + str(k_feat) + ".p", 'rb')
            results_hours = pickle.load(fin, encoding='latin1')
            fin.close()

            plot_results(results_hours,xticklabelsvals,all_predictors,result_dir + str(k_feat)+'/figures/ML_Results_cumulative_one_min_RR'+str(k_feat))

            plt.close('all')

        ####  a. Plot Risk Scores for different models
        for color, name in [
            ('b', 'EC'),
            # ('g', 'SL'),
            # ('r', 'LR'),
            # ('c', 'SK'),
            # ('m', 'RF')
        ]:
            risk_score_file_name=result_dir + str(70) + "/riskscores_"+name+".p"
            if os.path.isfile(risk_score_file_name):
                print(risk_score_file_name)
                plot_risk_scores(risk_score_file_name,name)
                break

        ####  b. Plot Risk Scores for EC classifier at different times
        name = "EC"
        name = "NN"
        # results_file_name = result_dir + str(70)+"/"
        results_file_name = result_dir + 'NN' + "/"
        for numofhours in np.arange(start_time, 168+12, 12):
            if numofhours >= 48:
                risk_score_file_name = results_file_name + "riskscores_"+name +"_"+str(numofhours)+ "__hourly.p"
                if os.path.isfile(risk_score_file_name):
                    print(risk_score_file_name)
                    plot_risk_scores(risk_score_file_name, name,vmin=0.25,vmax=0.8)
                    plt.savefig(risk_score_file_name[0:-2] + '_line_plot' '.png')
                    plt.close()
                    plt.savefig(risk_score_file_name[0:-2] + '_images' '.png')
                    plt.close()

        ### c. plot correct discharges over time for different models
        name="EC"
        results_file_name = result_dir + str(70)+"/"
        for numofhours in np.arange(0, 168+12, 12):
            if numofhours >= 48:
                risk_score_file_name = results_file_name + "riskscores_"+name +"_"+str(numofhours)+ "__hourly.p"
                if os.path.isfile(risk_score_file_name):
                    print(risk_score_file_name)
                    plot_tpv_fpv_over_time(risk_score_file_name)


