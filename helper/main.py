import numpy as np
import pyodbc
import os
import helper.util_func as uf
import helper.util_ml as uml

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import _pickle as pickle
import pandas as pd
import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def generate_risk_score(data):
    return

def plot_data_1(philips_resample_wide,keys,ncols = 2, show = True):

    temp = philips_resample_wide['timestamp']
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))

    for (key, ax) in zip(keys, axes.flatten()):
        ax.plot(temp,philips_resample_wide[key])
        philips_resample_wide[['timestamp',key]].plot(x='timestamp',y=key,ax=ax,label=key)
        # plt.title(key)
        # plt.ylabel(key)
    plt.tight_layout()

    if show:
        plt.show()
    return

def get_DCI_Model():

    # 4.2b : Retrieve Model
    # k_feats = 70
    # result_dir = 'models/results_NO_ICP_TEMP_strt_dci/'
    # #
    # fileName = result_dir + str(k_feats) + "/ML_Results_cumulative_one_min_RR_" + str(k_feats) + ".p"
    # results_hours = pickle.load(open(fileName, "rb"),encoding='latin1')
    # (classifier_max_auc, auc_val) = uml.get_max_classifier_before_dci(results_hours, classifier_name='EC',
    #                                                               classifier_at_dci=False, classifier_at_hour=-3.5,
    #                                                               toplot=False)
    # classifer_eval = classifier_max_auc
    # fout = open('models/DCI_Model.pickle', 'wb')
    # pickle.dump(classifer_eval, fout)

    classifier_eval = pickle.load(open('models/DCI_Model.pickle', "rb"), encoding='latin1')
    return classifier_eval

# the function will generate risk scores and save the results in the figures
# 1. The input to the function are
#       a. list of patients with their demographic information and
#       b. classifer
# 2. Output - None, generates the risk scores and saves it under figures

def generate_risk_scores(dem_patients = None, classifier_eval=None,var_ranges=None):
    demographic_vars = ['Mean_CONTINUOUS_Age', 'Mean_CATEGORICAL_Sex', 'Mean_CONTINUOUS_HH', 'Mean_CONTINUOUS_MFS',
                        'Mean_CONTINUOUS_WFNS', 'Mean_CONTINUOUS_GCS']
    keys = ['HR', 'AR-M', 'AR-D', 'AR-S', 'SPO2', 'RR']
    for index, pat in dem_patients.iterrows():
            mrn = pat['MRN']
            print(mrn)
            dem_feats_value = pat[demographic_vars]
            if dem_feats_value['Mean_CATEGORICAL_Sex'] == 'F':
                dem_feats_value['Mean_CATEGORICAL_Sex'] = 0
            else:
                dem_feats_value['Mean_CATEGORICAL_Sex'] = 1
            dem_feats_value = dem_feats_value.to_frame('Value').reset_index().rename(columns={'index': 'Features'})

            admndt = pat['Admin']
            dischdt = pat['Discharge']

            if not os.path.isfile('data/' + str(mrn) + '.xlsx'):  # get resampled data
                print('download data')
                if os.path.isfile('helper/data/' + str(mrn) + '.csv'):  # get 1 second data
                    philips = pd.read_csv('helper/data/' + str(mrn) + '.csv')
                    if 'timestamp' in philips.keys():
                        philips['timestamp'] = pd.to_datetime(philips['timestamp'])
                    else:
                        continue
                else:
                    continue  # commented below line as this is not required for Federated learning project
                    # with pyodbc.connect("DSN=Cloudera ODBC Driver for Impala", autocommit=True) as conn:
                    #     philips = pd.read_sql(query, conn)
                philips['sublabel'] = philips['sublabel'].replace({
                    'ARTm': 'AR-M',
                    'ABPm': 'AR-M',
                    'ARTd': 'AR-D',
                    'ABPd': 'AR-D',
                    'ARTs': 'AR-S',
                    'ABPs': 'AR-S',
                    'SpO₂': 'SPO2'  ##  (MM) This line was missing, this will ignore SPO2
                })
                if 'AR-M' not in np.unique(philips['sublabel']) or 'AR-D' not in np.unique(philips['sublabel']):
                    continue
                # downsample and save this data
                for k in keys:
                    philips.loc[
                        (philips['sublabel'] == k) & (philips['value'] > var_ranges.get(k)['max']), 'value'] = np.nan
                    philips.loc[
                        (philips['sublabel'] == k) & (philips['value'] <= var_ranges.get(k)['min']), 'value'] = np.nan
                # Step 4 : Downsample to a min
                # philips= philips.set_index('timestamp')
                philips_resample = philips.groupby(['sublabel']).resample('60s', on='timestamp').median().reset_index()
                philips_resample = philips_resample[philips_resample['sublabel'].isin(keys)]
                philips_resample.to_excel('helper/' + str(mrn) + '.xlsx')
                # Step 4.1 : Convert to wide format as we are using only wide format
                # philips1_resample_wide = philips_resample_wide
                # philips1_resample_wide.loc[(philips1_resample_wide['timestamp'] > '2021-12-25') & (philips1_resample_wide['timestamp'] < '2021-12-26'), keys] = np.nan
                # philips1_resample_wide = philips1_resample_wide.loc[philips1_resample_wide['timestamp'] < '2021-12-27']
            else:
                philips_resample = pd.read_excel(('data/' + str(mrn) + '.xlsx'))

            philips_resample_wide = philips_resample.pivot(index='timestamp', columns='sublabel',
                                                           values='value').reset_index()
            philips_resample_wide['mrn'] = mrn
            philips_resample_wide = philips_resample_wide.ffill()  # forward fill missing data

            # Step 2: Create feaatures ( xcorr feats)
            pat_predictor_vals = uml.compute_xcorr_feats_Impala(philips_resample_wide, dem_feats_value,
                                                                keys=keys)
            # Generate Risk Scores
            compute_hours = 1  # this give risk scores hourly, if we change it to 12 then it will be one value every 12 hours
            (probab_vals, timevals1) = uml.compute_RiskScores_Impala(classifier_eval, pat_predictor_vals, 'EC', keys=keys,
                                                                     include_dems=True, compute_hours=compute_hours)
            # philips1 = philips_resample[philips_resample['sublabel'].isin(keys)]
            plot_data_1(philips_resample_wide, keys, ncols=3)
            plt.savefig('figures/' + str(mrn) + '_raw_data.png')
            plt.close('all')

            # plt.style.use('dark_background')
            plt.figure()
            plt.plot(timevals1, probab_vals, 'o-')
            plt.ylim([0.1, 1])
            plt.title('Risk Score')
            plt.xlabel('Time')
            plt.ylabel('Risk Scores')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.axhline(y=0.35, color='red', linestyle='--')
            plt.savefig('figures/' + str(mrn) + '_Risk_score.png')
            plt.close('all')


if __name__ == '__main__':
    print('hello')

    # stpe 0 : Get DCI Model
    classifier_eval = get_DCI_Model()
    # step 1: get patient
    dem_patients = pd.read_excel('helper/CurrentPatientDemographic.xlsx')
    dem_patients=dem_patients[~np.isnat(dem_patients['Admin'])]
    dem_patients=dem_patients[~np.isnat(dem_patients['Discharge'])]
    dem_patients.rename(columns=
                        {'Age':'Mean_CONTINUOUS_Age',
                         'HH':'Mean_CONTINUOUS_HH',
                         'mFS':'Mean_CONTINUOUS_MFS',
                         'Sex':'Mean_CATEGORICAL_Sex',
                         'WFNS':'Mean_CONTINUOUS_WFNS',
                          'GCS':'Mean_CONTINUOUS_GCS'},inplace=True)

    # Step 1: Download all the data for the prospective patients
    var_ranges_file = 'helper/Variable_Range.csv'
    var_ranges = uf.get_min_max_var(var_ranges_file)

    #step 2 : Generate risk scores and save it in fiigures
    generate_risk_scores(dem_patients,classifier_eval=classifier_eval)


