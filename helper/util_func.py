
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import csv, os
import _pickle as pickle
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
#import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This function aggregates the different time series corresponding to CVP.
def get_cvp(row):
    if 'CVP-M' in row:
       if row['CVP-M'] != ' ' and row['CVP-M'] != 'NA': return row['CVP-M']
    if 'CVP1' in row:
        if row['CVP1'] != ' ' and row['CVP1'] != 'NA': return row['CVP1']
    if 'CVP2' in row:
        if row['CVP2'] != ' ' and row['CVP2'] != 'NA': return row['CVP2']
    if 'CVP3' in row:
        if row['CVP3'] != ' ' and row['CVP3'] != 'NA': return row['CVP3']
    if 'CVP4' in row:
        if row['CVP4'] != ' ' and row['CVP4'] != 'NA': return row['CVP4']
    if 'CVP5' in row:
        if row['CVP5'] != ' ' and row['CVP5'] != 'NA': return row['CVP5']
    if 'CVP6' in row:
        if row['CVP6'] != ' ' and row['CVP6'] != 'NA': return row['CVP6']
    if 'CVP7' in row:
        if row['CVP7'] != ' ' and row['CVP7'] != 'NA': return row['CVP7']
    if 'CVP8' in row:
        if row['CVP8'] != ' ' and row['CVP8'] != 'NA': return row['CVP8']
    return ' '

def get_arm(row):
    if 'AR1-M' in row:
        if row['AR1-M'] != ' ' and row['AR1-M'] != 'NA': return row['AR1-M']
    if 'AR2-M' in row:
        if row['AR2-M'] != ' ' and row['AR2-M'] != 'NA': return row['AR2-M']
    if 'AR3-M' in row:
        if row['AR3-M'] != ' ' and row['AR3-M'] != 'NA': return row['AR3-M']
    if 'AR4-M' in row:
        if row['AR4-M'] != ' ' and row['AR4-M'] != 'NA': return row['AR4-M']
    if 'AR5-M' in row:
        if row['AR5-M'] != ' ' and row['AR5-M'] != 'NA': return row['AR5-M']
    if 'AR6-M' in row:
        if row['AR6-M'] != ' ' and row['AR6-M'] != 'NA': return row['AR6-M']
    if 'AR7-M' in row:
        if row['AR7-M'] != ' ' and row['AR7-M'] != 'NA': return row['AR7-M']
    if 'AR8-M' in row:
        if row['AR8-M'] != ' ' and row['AR8-M'] != 'NA': return row['AR8-M']
    return ' '

def get_ars(row):
    if 'AR1-S' in row:
        if row['AR1-S'] != ' ' and row['AR1-S'] != 'NA': return row['AR1-S']
    if 'AR2-S' in row:
        if row['AR2-S'] != ' ' and row['AR2-S'] != 'NA': return row['AR2-S']
    if 'AR3-S' in row:
        if row['AR3-S'] != ' ' and row['AR3-S'] != 'NA': return row['AR3-S']
    if 'AR4-S' in row:
        if row['AR4-S'] != ' ' and row['AR4-S'] != 'NA': return row['AR4-S']
    if 'AR5-S' in row:
        if row['AR5-S'] != ' ' and row['AR5-S'] != 'NA': return row['AR5-S']
    if 'AR6-S' in row:
        if row['AR6-S'] != ' ' and row['AR6-S'] != 'NA': return row['AR6-S']
    if 'AR7-S' in row:
        if row['AR7-S'] != ' ' and row['AR7-S'] != 'NA': return row['AR7-S']
    if 'AR8-S' in row:
        if row['AR8-S'] != ' ' and row['AR8-S'] != 'NA': return row['AR8-S']
    return ' '

def get_ard(row):
    if 'AR1-D' in row:
        if row['AR1-D'] != ' ' and row['AR1-D'] != 'NA': return row['AR1-D']
    if 'AR2-D' in row:
        if row['AR2-D'] != ' ' and row['AR2-D'] != 'NA': return row['AR2-D']
    if 'AR3-D' in row:
        if row['AR3-D'] != ' ' and row['AR3-D'] != 'NA': return row['AR3-D']
    if 'AR4-D' in row:
        if row['AR4-D'] != ' ' and row['AR4-D'] != 'NA': return row['AR4-D']
    if 'AR5-D' in row:
        if row['AR5-D'] != ' ' and row['AR5-D'] != 'NA': return row['AR5-D']
    if 'AR6-D' in row:
        if row['AR6-D'] != ' ' and row['AR6-D'] != 'NA': return row['AR6-D']
    if 'AR7-D' in row:
        if row['AR7-D'] != ' ' and row['AR7-D'] != 'NA': return row['AR7-D']
    if 'AR8-D' in row:
        if row['AR8-D'] != ' ' and row['AR8-D'] != 'NA': return row['AR8-D']
    return ' '

def get_cpp(row):
    if 'CPP1' in row:
        if row['CPP1'] != ' ' and row['CPP1'] != 'NA': return row['CPP1']
    if 'CPP2' in row:
        if row['CPP2'] != ' ' and row['CPP2'] != 'NA': return row['CPP2']
    if 'CPP3' in row:
        if row['CPP3'] != ' ' and row['CPP3'] != 'NA': return row['CPP3']
    if 'CPP4' in row:
        if row['CPP4'] != ' ' and row['CPP4'] != 'NA': return row['CPP4']
    if 'CPP5' in row:
        if row['CPP5'] != ' ' and row['CPP5'] != 'NA': return row['CPP5']
    if 'CPP6' in row:
        if row['CPP6'] != ' ' and row['CPP6'] != 'NA': return row['CPP6']
    if 'CPP7' in row:
        if row['CPP7'] != ' ' and row['CPP7'] != 'NA': return row['CPP7']
    if 'CPP8' in row:
        if row['CPP8'] != ' ' and row['CPP8'] != 'NA': return row['CPP8']
    return ' '

#
# def get_labs():

# This function loads a CSV file.
def load_csv(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def load_h5_data(h5dir,var_ranges,keys,filName = '2099_5251902_Cybulsky'):
    patient = defaultdict(lambda: [])
    patdir = os.path.join(h5dir, filName)
    addTime = 1
    for key in keys:
        for filename in os.listdir(patdir):
            if filename.endswith(".h5"):
                p1 = load_h5(os.path.join(patdir,filename),'VS',key,var_ranges)
                if p1 is None:
                    continue
                if len(p1[key]) > 0:
                    patient[key] = np.append(patient[key],p1[key])
                else:
                    patient[key] = np.append(patient[key],np.ones(p1['TimeVector'].shape[1])*np.nan)
                if addTime or len(patient['TimeVector'])==0:
                        patient['TimeVector'] = np.append(patient['TimeVector'], p1['TimeVector'])
        addTime = 0
            # patient.update(p1)
    return patient


def load_h5(h5fileName,type,label,var_ranges):
    patient = defaultdict(lambda: {})
    if type=='WaveForm':
        groupName = '/RAW_Data/STP_Data/WaveForm_Data/';
        attribname = 'Channel'
    else:
        groupName = '/RAW_Data/STP_Data/VS_Data/';
        attribname = 'Param'
    f1 = h5py.File(h5fileName, 'r')
    if not (groupName in f1):
        return
    f = f1[groupName]
    timeVector = []
    data = defaultdict(lambda: {})
    indx = 0;
    for k in f.keys():
        if indx == 0:
            timeVector = f.get(k)
            timeVector = np.array(timeVector)
        else:
            attribval = f[k].attrs[attribname].decode("utf-8")
            # print(attribval)
            boolval = check_data(attribval,label)
            if boolval:
                data[attribval] = np.array(f.get(k))
        indx = indx+1
    f1.close()

    ## consolidate the data ( all the data should
    patient['TimeVector'] = np.asarray(timeVector).reshape(1, -1)
    if len(data) == 0:
        return patient
    tempdata =[]
    for k1 in data.keys():
        t1 = data[k1]
        if len(t1) > 0 :
            if var_ranges.get(label):
                t1[np.where(t1 > var_ranges.get(label)['max'])] = np.NAN
                t1[np.where(t1 <= var_ranges.get(label)['min'])] = np.NAN
            tempdata.append(t1)
    # tempdata = np.nanmean(tempdata,0) ---- Since the data can be of different lengths we cannot take mean, for e.g. AR1-M and AR2-M can be of different lengths

    # tempdata1 = np.ones((np.max([len(ps) for ps in tempdata]), len(tempdata))) * np.nan  # define empty array
    tempdata1 = np.ones((len(timeVector), len(tempdata))) * np.nan  # define empty array - instead of the longest array in the dataset use the timevector as that is the longest
    for i, c in enumerate(tempdata):  # populate columns
        tempdata1[:len(c), i] = c.reshape(1, -1)
    tempdata =np.nanmean(tempdata1, 1)

    patient[label] =   np.asarray(tempdata).reshape(1,-1)


    # for i in range(len(tempdata)):
    #     timeval = matlab_to_python_datetime(timeVector[i, 0])
    #     timeval = time.mktime(timeval.timetuple())
    #     patient[label][timeval] = tempdata[i]

    return patient

def matlab_to_python_datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    # import time
    # print(timeval)
    # print(time.mktime(timeval.timetuple()))
    # print(datetime.fromtimestamp(time.mktime(timeval.timetuple())))


def roundTime(dt=None, roundTo=60, to='up'):
    """Round a datetime object to any time laps in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.

    """
    if dt == None: dt = datetime.datetime.now()
    seconds = (dt - dt.min).seconds
    #print(seconds)
    # // is a floor division, not a comment on following line:

    if to == 'up':
        # // is a floor division, not a comment on following line (like in javascript):
        rounding = (seconds + roundTo) // roundTo * roundTo
    elif to == 'down':
        rounding = seconds // roundTo * roundTo
    else:
        rounding = (seconds + roundTo / 2) // roundTo * roundTo

    # rounding = (seconds+roundTo/2) // roundTo * roundTo
    # print(rounding)
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)

def check_data(attribValue, labels):
    boolret = 0
    if labels == 'AR-S':
        if attribValue == 'AR1-S' or attribValue == 'AR2-S'  or attribValue == 'AR3-S' or attribValue == 'AR4-S' or attribValue == 'AR5-S' or attribValue == 'AR6-S' :
            boolret = 1
    elif labels == 'AR-D':
        if attribValue == 'AR1-D' or attribValue == 'AR2-D'  or attribValue == 'AR3-D' or attribValue == 'AR4-D' or attribValue == 'AR5-D' or attribValue == 'AR6-D' :
            boolret = 1
    elif labels == 'AR-M':
        if attribValue == 'AR1-M' or attribValue == 'AR2-M'  or attribValue == 'AR3-M' or attribValue == 'AR4-M' or attribValue == 'AR5-M' or attribValue == 'AR6-M' :
            boolret = 1
    elif labels == 'ICP':
        if attribValue == 'IC1' or attribValue == 'IC3'  or attribValue == 'ICP1' or attribValue == 'ICP2' or attribValue == 'ICP3' or attribValue == 'ICP4' or attribValue == 'ICP5' or attribValue == 'ICP6' or attribValue == 'ICP7' or attribValue == 'ICP8' :
            boolret = 1
    elif labels == 'SPO2':
        if attribValue == 'SPO2-%' or attribValue == 'SPO2':
            boolret = 1
    elif labels == 'TEMP':
        if attribValue == 'TMP1' or attribValue == 'TMP2' or attribValue == 'TEMP':
            boolret = 1
    elif labels == 'RR':
        if attribValue == 'RRGE' or attribValue == 'RESP' or attribValue == 'RR':
            boolret = 1
    elif labels == attribValue:
        boolret = 1
    return boolret

    #
    #     print(f1[('/RAW_Data/STP_Data/WaveForm_Data/' + k)].attrs['Channel'])  # get channel name
    #
    # plt.hold(1)
    # for k in f.keys():
    #
    #     if k1 >= 1:
    #         print(k)
    #         n1 = f1.get('/RAW_Data/STP_Data/WaveForm_Data/' + k);
    #         print(np.array(n1).shape)
    #         plt.plot(np.array(n1))
    #         # n1 = f1.get([('/RAW_Data/STP_Data/WaveForm_Data/'+ k )]);
    #         # print(f1[('/RAW_Data/STP_Data/WaveForm_Data/' + k)])  # get channel name
    #         print(f1[('/RAW_Data/STP_Data/WaveForm_Data/' + k)].attrs['Channel'])  # get channel name
    #     k1 = k1 + 1
    # plt.show()


# This function loads the clinical data.
# It return a dictionary of (ID -> patient).
# Each patient is a dictionary where keys are the variable names.
def load_clinical_data(filename):
    data = load_csv(filename)
    patients = {}
    for row in data:
        pid = int(row['ID'])
        patients[pid] = row
    return patients

# This function loads the physiological time series corresponding to one patient.
# It returns a dictionary where keys are the variables names.
# Each physio time series is also stored as a dictionary: (time -> value).
def load_physio(filename,keys):
    data = load_csv(filename)
    patient = defaultdict(lambda: {})
    for row in data:
        if (row['ID']) == '': continue;
        pid = int(row['ID'])
        partime = float(row['ParTime'])
        row['CVP'] = get_cvp(row)
        row['AR-M'] = get_arm(row)
        row['AR-S'] = get_ars(row)
        row['AR-D'] = get_ard(row)
        # print(row['Real_Time'])
        datetime_object = datetime.strptime(row['Real_Time'], '%m/%d/%Y %H:%M:%S')
        # print(datetime_object)
        # patient['Real_Time'][partime] = datetime_object
        for k in keys:
            if k in data[0]:
                if row[k] != ' ' and row[k] != 'NA' and row[k] != '':
                    patient[k][partime] = float(row[k])
                    patient['Real_Time_'+k][partime] = datetime_object
    return patient





# This function transforms the physio time series stored as dictionary (time -> value)
# into physio time series stored as numpy array. Each index of the array corresponds
# to one minute, and data from admission to discharge date kept, by default.
# Moreover, the data can be normalized normalized (mean 0 and variance 1), and missing data are imputed
# by the mean. This function also saves correlations between HR and other keys in the list
def extract_variables_h5(patient, metameddata, metalabdata, keys,ndays=4,normalize=False,aggregate=False,daystolookaroundachnor = 7):
    p = {}
    # Times series are sampled every 5 seconds, so we have 12 points per min.
    # Thus, each day of data corresponds to 12 * 60 * 24 points.

    # if patient['Anchor_date'] =='#N/A':
    #     return
    if patient['Bleed_date'] =='#N/A':
        return
    p ['Bleed_date'] = patient['Bleed_date']

    # anchor_date = datetime.strptime(patient['Bleed_date'], '%m/%d/%Y %H:%M') # we anchor it with the bleed date
    # starttime = anchor_date # anchor date will be the start date
    # endtime = anchor_date+timedelta(days=2*daystolookaroundachnor) ; # get 10 or 14 days after the bleed date

    anchor_date = datetime.strptime(patient['Consolidated Admission Date'], '%m/%d/%Y %H:%M') # we anchor it with the admission date
    starttime = anchor_date # anchor date will be the start date
    endtime = datetime.strptime(patient['Consolidated DischargeDate'],
                                    '%m/%d/%Y %H:%M')  # we anchor it with the bleed date
    temp = (endtime-starttime)
    npoints = int(30 * temp / timedelta(minutes=1)) # since we have 30 points in a minute
    # npoints = 30 * 60 * 24 * (daystolookaroundachnor * 2)  # we have point every 2 sec(h5) so in one min have 30 points
    for name in keys:
        y = np.empty(npoints)
        y[:] = np.NAN
        if name in patient:
            # mean = np.mean(list(patient[name].values()))
            # std = np.std(list(patient[name].values()))
            mean = np.nanmean(list(patient[name]))
            std = np.nanstd(list(patient[name]))

            if std == 0.0:
                std = 1.0
            for t, v in zip(patient['TimeVector'],patient[name]):
                # t = a[0];v = a[1]
                # t1 = b[0];realtime = b[1]
                realtime = roundTime(matlab_to_python_datetime(t),1)
                # timeval = time.mktime(timeval.timetuple())
                # if var_ranges.get(name):
                #     if v<var_ranges[name]['min'] or v>var_ranges[name]['max']:
                #         v = np.NAN
                # this code is to get the data before and after ANCHOR TIME


                if realtime>=starttime and realtime < endtime:
                    if normalize:
                        y[int((realtime-starttime).total_seconds()/2)] = (v - mean) / std

                        # zscore = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
                        # plt.plot(np.round(zscore(p['AR-S']))); to take zscore and round it to nearest integer
                    else:
                        # print(realtime)
                        # print(starttime)
                        # print(int((realtime - starttime).total_seconds() / 5))
                        y[int((realtime-starttime).total_seconds()/2)] = np.nanmean([y[int((realtime-starttime).total_seconds()/2)] ,v])



                # this code is for getting the data from time of ICTUS
                # if t/5 < npoints:
                #     if normalize:
                #         x[int(t/5)] = (v - mean) / std
                #     else:
                #         x[int(t / 5)] = v
        if aggregate:

            # We aggregate data to get one point per minute (instead of 12).
            # p[name] = np.nanmedian(x.reshape([-1, 12]), axis=1)
            p[name] = np.nanmedian(y.reshape([-1, 30]),axis=1) # # average over 30 points for H5 since we have a data point every 2 secs, for G.E we will have value every 2 sec, but philips it will be every one sec, and rest of the values will be Nan so this will work, since y is intialized with nan values
        else:
            p[name] = y;

    for key in patient:  ## save all the non array values - demographics and dates
        if isinstance(patient[key], np.ndarray):
            continue
        else:
            p[key] = patient[key]
    # save med data
    # for idx, key in metameddata['Med_names'].iteritems():
    #     p[key] = patient[key]
    #
    # for idx, key in metalabdata['Name'].iteritems():
    #     p[key] = patient[key]

    ## compute cross correlation here
    p  = compute_cross_corr(p,keys)
    return p


def compute_cross_corr(p,keys,lag=1,window_duration =10,min_data_lim = 0.5):
    p1 = pd.DataFrame.from_dict(p)
    ## initialzie xcorr values  to nan
    for k in keys:
        if k == 'HR':
            continue
        p1['max_Xcorr_HR_'+k] = np.nan
        p1['min_Xcorr_HR_' + k] = np.nan
    for i  in np.arange(window_duration-1,p1.shape[0]): # loop through 10 mins window with an increment of 1 min - since the data is sample every min so increment is 1
        # print(p1.iloc(i,i+10))
        df_new = p1.iloc[i - window_duration + 1:i+1]  # window of 10 mins, since the data is already sampled every minute
        a = df_new['HR'] # get HR
        if np.sum(~a.isna()) < min_data_lim*len(a): # if total number of nan values is less than the minimum amount of data then do not process
            continue
        for k in keys:
            # print(k)
            if k == 'HR' or k == 'Bleed_date':  # no need to do corr with HR
                continue
            if k not in df_new.keys():
                continue
            b = df_new[k].shift(lag).astype(float)# lag of 1
            if np.sum(~np.isnan(b)) < min_data_lim * len(b):  # Compute this only when there is minimum amount of data
                continue
            corr = np.correlate(a - np.nanmean(a), b - np.nanmean(b), mode='full')/ (np.nanstd(a) * np.nanstd(b) * len(a))

            # plt.figure()
            # plt.plot(corr / (np.nanstd(a) * np.nanstd(b) * len(df_new['HR'])))
            # plt.title(k)
            # print(np.nanmax(corr / (np.nanstd(a) * np.nanstd(b) * len(df_new['HR']))))
            p1.loc[i, 'max_Xcorr_HR_'+k] = np.nanmax(corr) ##Save correlation values - both Max and Min
            p1.loc[i, 'min_Xcorr_HR_' + k] = np.nanmin(corr)
    return p1




# This function transforms the Medical data as time series stored as dictionary (time -> value)
# into physio time series stored as numpy array. Each index of the array corresponds
# to one minute,
def extract_med_data(patient, medfileName, metameddata, ndays=4,normalize=False,aggregate=False,daystolookaroundachnor = 7):
    data = pd.read_csv(medfileName)
    # patient = defaultdict(lambda: {})
    data['TASKPERFORMEDFROMDTM'] = pd.to_datetime(data['TASKPERFORMEDFROMDTM'],
                                                           format='%Y-%m-%d-%H.%M.%S.%f')
    # remove drips
    idx = data[data.keys()[11:]] # med trypes start from 11 ( index 10 is drips)
    data = data[idx.sum(axis=1) > 0]


    if patient['Bleed_date'] =='#N/A':
        return
    # anchor_date = datetime.strptime(patient['Anchor_date'], '%m/%d/%Y %H:%M')
    # starttime = anchor_date-timedelta(days=daystolookaroundachnor)
    # endtime = anchor_date+timedelta(days=daystolookaroundachnor)

    anchor_date = datetime.strptime(patient['Bleed_date'], '%m/%d/%Y %H:%M') # we anchor it with the bleed date
    starttime = anchor_date # anchor date will be the start date
    endtime = anchor_date+timedelta(days=2*daystolookaroundachnor) ; # get 10 or 14 days after the bleed date
    npoints = 60 * 24 * (daystolookaroundachnor * 2)  # Get time points on a grid starting spaced every min starting from bleed date to 14 days after bleed date total number of points will be 60 (mins) * 24(hours)*14(days)
    for idx, key in metameddata['Med_names'].iteritems():
        # print(key)
        filt_med_data = data[data['MedType'] == key] # get values for the given medicaiton
        filt_med_data = filt_med_data[filt_med_data['TASKPERFORMEDFROMDTM'] >= starttime] # get data between start and end time
        filt_med_data = filt_med_data[
            filt_med_data['TASKPERFORMEDFROMDTM'] < endtime]  # get data between start and end time
        y = np.zeros(npoints, dtype=bool)
        for index, row in filt_med_data.iterrows():
            # print(row['TASKPERFORMEDFROMDTM'], row['MedType'])
            # print(starttime)
            # print(row['TASKPERFORMEDFROMDTM'])
            y[int((row['TASKPERFORMEDFROMDTM'] - starttime).total_seconds()/60)] = True
        patient[key] =  y

    return patient



# This function transforms the Medical data as time series stored as dictionary (time -> value)
# into physio time series stored as numpy array. Each index of the array corresponds
# to one minute,
def extract_lab_data(patient, labfileName, metalabdata, ndays=4,normalize=False,aggregate=False,daystolookaroundachnor = 7):
    data = pd.read_csv(labfileName)
    # patient = defaultdict(lambda: {})
    data['PRIMARY_TIME'] = pd.to_datetime(data['PRIMARY_TIME'],
                                                           format='%Y-%m-%d-%H.%M.%S.%f')
    if patient['Bleed_date'] =='#N/A':
        return

    anchor_date = datetime.strptime(patient['Bleed_date'], '%m/%d/%Y %H:%M') # we anchor it with the bleed date
    starttime = anchor_date # anchor date will be the start date
    endtime = anchor_date+timedelta(days=2*daystolookaroundachnor) ; # get 10 or 14 days after the bleed date
    npoints = 60 * 24 * (daystolookaroundachnor * 2)  # Get time points on a grid starting spaced every min starting from bleed date to 14 days after bleed date total number of points will be 60 (mins) * 24(hours)*14(days)
    for idx, key in metalabdata['Name'].iteritems():
        # print(key)
        filt_lab_data = data[data['LAB_NAME_TYPE'] == key] # get values for the given medicaiton
        filt_lab_data = filt_lab_data[filt_lab_data['PRIMARY_TIME'] >= starttime] # get data between start and end time
        filt_lab_data = filt_lab_data[
            filt_lab_data['PRIMARY_TIME'] < endtime]  # get data between start and end time
        y = np.empty(npoints)
        y[:] = np.NAN
        for index, row in filt_lab_data.iterrows():
            # print(row['TASKPERFORMEDFROMDTM'], row['MedType'])
            # print(starttime)
            # print(row['TASKPERFORMEDFROMDTM'])
            y[int((row['PRIMARY_TIME'] - starttime).total_seconds()/60)] = row['NUM_VALUE']
        patient[key] =  y

    return patient


# This function returns the max amd min values for variables
def get_min_max_var(var_ranges_file):
    var_ranges = load_csv(var_ranges_file)
    minmax = defaultdict(lambda: {})
    for row in var_ranges:
        var_name = (row['Features'])
        minmax[var_name]["min"] = int(row['Min Value'])
        minmax[var_name]["max"] = int(row['Max Value'])
    return minmax

def is_number(n):
    try:
        float(n)   # Type-casting the string to `float`.
                   # If string is not a valid `float`,
                   # it'll raise `ValueError` exception
    except (ValueError,TypeError):
        return False
    return True

def format_date(date_val):
    try:
        date_ret = datetime.strptime(date_val, '%m/%d/%Y %H:%M') # it'll raise `ValueError` exception if hours and mins are missing
    except ValueError:
        date_ret = datetime.strptime(date_val,
                                     '%m/%d/%Y')
    return date_ret

# summarizes the data hourly
# Step 1. : As per Fairchild paper, computes mean,std, median in 10 mins window
# Step 2 : Take the 60 mins avearage (6 points)
# Step 3 : Anchor data to the Anchor date : DCI Date for +ve and 7 days for DCI- patients
def summarize_hourly(file_name,anchor_date,admin_field_name = 'Consolidated Admission Date',days_around_anchor = 7,anchordata=True,dci_index = 'DCI_Index',bleed_day_name = 'Bleed Day'):
    fin = open(file_name, 'rb')
    patient = pickle.load(fin, encoding='latin1')
    fin.close()


    if anchor_date=='NA': # Compute Anchor date for UTH data
        anchor_date = np.unique(patient['Bleed Day'])[0] + ' 14:00'
        anchor_date = datetime.strptime(anchor_date, '%m/%d/%Y %H:%M')
        if int(np.unique(patient[dci_index])[0]) == 1:
            days_to_add = int(np.unique(patient['Clinical DCI Bleed Date'])[0])
            if np.unique(patient['Angiographic Vasospasm Bleed Date']) != '':
                angio_vsp_dt = int(np.unique(patient['Angiographic Vasospasm Bleed Date'])[0]);
                if days_to_add>angio_vsp_dt:
                    days_to_add = angio_vsp_dt
            # Anchor around 2 PM as the adjudication of UTH data is at 2 PM
            # time_to_add = timedelta(days=int(np.unique(patient['Clinical DCI Bleed Date'])[0]))
            time_to_add = timedelta(days=days_to_add)
        else:
            time_to_add = timedelta(days = 7)
        anchor_date = anchor_date +  time_to_add



    # take hourly average
    patient_hourly_vals = pd.DataFrame()

    for key in sorted(patient.keys()):
        # loop through keys remove dates and take the mean
       if not is_number(patient[key][0]):
       # if 'DATE' in key.upper() or 'DOB' in key.upper() or 'NAME' in key.upper() or 'MURAD' in key.upper() :
           patient_hourly_vals[key] = [patient[key][0]] * patient_hourly_vals.shape[0]
       else:
            y = patient[key].astype(float).values
            nanstappend = int(10-np.mod(len(y),10)) # taking 10 mins mean,SD, Median, so make sure that the length is divisible by 10 if not append nans to the end and then take average
            y_nans = np.empty(nanstappend)
            y_nans[:] = np.NAN
            y_append = np.append(y,y_nans)
            for func,name in zip([np.nanmean, np.nanmedian,np.nanstd],['Mean','Median','STD']):
                y_10_func = func(y_append.reshape([-1, 10]),axis=1)

                # take mean over 60 mins
                nans_to_append_hour = int(6-np.mod(len(y_10_func),6))  # taking 60 mins (6 values of 10 mins ) mean,SD, Median, so make sure that the length is divisible by 10 if not append nans to the end and then take average
                y_nans_1 = np.empty(nans_to_append_hour)
                y_nans_1[:] = np.NAN
                y_10_func_append = np.append(y_10_func, y_nans_1)
                y_60_key = np.nanmean(y_10_func_append.reshape([-1, 6]), axis=1)
                patient_hourly_vals[name+'_'+key] = y_60_key

    ## Anchor Data to anchor date
    if anchordata:
        # 7 days before and after anchor
        total_vals = 2* days_around_anchor *  24
        y_nans = np.empty(total_vals)
        y_nans[:]=np.NAN

        y_anchored = pd.DataFrame(columns=patient_hourly_vals.keys(),index=np.arange(0,total_vals))

        # admission_date = datetime.strptime(np.unique(patient[admin_field_name])[0], '%m/%d/%Y %H:%M')
        admission_date = format_date(np.unique(patient[admin_field_name])[0])
        start_idx = int(total_vals / 2 - int((anchor_date - admission_date) / timedelta(minutes=60)))
        if start_idx<0:
            start_idx_total = int(np.round((anchor_date - admission_date) / timedelta(minutes=60) - total_vals / 2))
            start_idx =0
        else:
            start_idx_total = 0

        if total_vals-patient_hourly_vals.shape[0]>0: # fewer than 14 days worth of data
            end_idx = int(start_idx+patient_hourly_vals.shape[0]-start_idx_total)
            end_idx_total_data = patient_hourly_vals.shape[0]
        else:
            end_idx = int(total_vals)
            end_idx_total_data = start_idx_total + total_vals -start_idx -1

        # y_nans[start_idx:end_idx] = patient_hourly_vals['Median_AR-D'].loc[start_idx_total:end_idx_total_data].values
        data_numpy=patient_hourly_vals.loc[start_idx_total:end_idx_total_data].values
        y_anchored.loc[start_idx:end_idx]  = pd.DataFrame(data_numpy,columns=patient_hourly_vals.keys(),index=np.arange(start_idx,start_idx+data_numpy.shape[0]))
        # data_numpy.shape





                #
                # plt.subplot(211)
                # plt.plot(patient_hourly_vals['Median_AR-D'])
                #
                # plt.subplot(212)
                # plt.plot(y_anchored['Median_AR-D'])




        return y_anchored
    else:
        admission_date = format_date(np.unique(patient[admin_field_name])[0])
        bleed_day = np.unique(patient[bleed_day_name])[0]
        try:
            bleed_day = datetime.strptime(bleed_day, '%m/%d/%Y')
        except:
            bleed_day = datetime.strptime(bleed_day, '%m/%d/%Y %H:%M')
        # return (patient_hourly_vals ,(admission_date-bleed_day),np.unique(patient['Clinical DCI Bleed Date'])[0])# data not anchored
    return (patient_hourly_vals, (admission_date - bleed_day),
            anchor_date-bleed_day)  # data not anchored



# summarizes the data hourly
# Step 1. : As per Fairchild paper, computes mean,std, median in 10 mins window
# Step 2 : Take the 60 mins avearage (6 points)
# Step 3 : Anchor data to the Anchor date : DCI Date for +ve and 7 days for DCI- patients
def summarize_hourly_Aachen(file_name,anchor_date,admin_field_name = 'Consolidated Admission Date',days_around_anchor = 7,anchordata = True):
    fin = open(file_name, 'rb')
    patient = pickle.load(fin, encoding='latin1')
    fin.close()

    # take hourly average
    patient_hourly_vals = pd.DataFrame()

    for key in sorted(patient.keys()):
        print(key)
        # loop through keys remove dates and take the mean
        if not is_number(patient[key][0]):
            # if 'DATE' in key.upper() or 'DOB' in key.upper() or 'NAME' in key.upper() or 'MURAD' in key.upper() :
            patient_hourly_vals[key] = [patient[key][0]] * patient_hourly_vals.shape[0]
        else:
            y = patient[key].astype(float).values
            nanstappend = int(10-np.mod(len(y),10)) # taking 10 mins mean,SD, Median, so make sure that the length is divisible by 10 if not append nans to the end and then take average
            y_nans = np.empty(nanstappend)
            y_nans[:] = np.NAN
            y_append = np.append(y,y_nans)
            for func,name in zip([np.nanmean, np.nanmedian,np.nanstd],['Mean','Median','STD']):
                y_10_func = func(y_append.reshape([-1, 10]),axis=1)
                # take mean over 60 mins
                nans_to_append_hour = int(6-np.mod(len(y_10_func),6))  # taking 60 mins (6 values of 10 mins ) mean,SD, Median, so make sure that the length is divisible by 10 if not append nans to the end and then take average
                y_nans_1 = np.empty(nans_to_append_hour)
                y_nans_1[:] = np.NAN
                y_10_func_append = np.append(y_10_func, y_nans_1)
                y_60_key = np.nanmean(y_10_func_append.reshape([-1, 6]), axis=1)
                patient_hourly_vals[name+'_'+key] = y_60_key



    ## Anchor Data to anchor date
    if anchordata:
        # 7 days before and after anchor
        total_vals = 2* days_around_anchor *  24
        y_nans = np.empty(total_vals)
        y_nans[:]=np.NAN

        y_anchored = pd.DataFrame(columns=patient_hourly_vals.keys(),index=np.arange(0,total_vals))




        # get DCI index
        dci_index = np.unique(patient['DCI_Index'])
        data_start_day = np.unique(patient['Data_start_day'])
        if dci_index==0: # TODO: CHECK IF ASSUMPTION IS CORRECTION : DAY 0 is bleed day

            start_id_anchor = int(data_start_day*24)
            start_idx_data = 0

            end_id_anchor = start_id_anchor + patient_hourly_vals.shape[0]
            end_idx_data = total_vals
            if end_id_anchor> total_vals:
                end_id_anchor=total_vals
        else:

            data_start_time = patient['Temp_time_val'][0] + timedelta(int(data_start_day[0]))
            patient.keys()

            dci_time = datetime.strptime('01/01/1900' + ' ' + str(patient['DCI time post ictus'][0])[:5], '%m/%d/%Y %H:%M') + timedelta(days=patient['DCI day post ictus'][0])
            anchor_hour = np.int(np.ceil((dci_time-data_start_time).total_seconds()/(60*60)))


            start_id_anchor = int(total_vals/2-anchor_hour)
            start_idx_data = 0
            end_id_anchor = start_id_anchor + patient_hourly_vals.shape[0]
            end_idx_data = total_vals
            if end_id_anchor> total_vals:
                end_id_anchor=total_vals
        # y_nans[start_idx:end_idx] = patient_hourly_vals['Median_AR-D'].loc[start_idx_total:end_idx_total_data].values
        data_numpy=patient_hourly_vals.loc[start_idx_data:end_idx_data].values
        y_anchored.loc[start_id_anchor:end_id_anchor]  = pd.DataFrame(data_numpy,columns=patient_hourly_vals.keys(),index=np.arange(start_id_anchor,start_id_anchor+data_numpy.shape[0]))
        # data_numpy.shape

        # plt.plot(y_anchored['Mean_AR-D'])
        # plt.xlim([1 , 336])
        # plt.axvline(x=168, color='k', linestyle='-', linewidth=2.5)

        # plt.subplot(211)
                # plt.plot(patient_hourly_vals['Median_AR-D'])

                # plt.subplot(212)
                # plt.plot(y_anchored['Median_AR-D'])




        return y_anchored
    else:
        data_start_day = np.unique(patient['Data_start_day'])

        dci_index = np.unique(patient['DCI_Index'])
        if dci_index == 0:
            dci_dt_wrt_bd = ''
        else:
            dci_dt_wrt_bd = timedelta(days = patient['DCI day post ictus'][0],hours=int(str(patient['DCI time post ictus'][0])[:2]))/timedelta(days=1)

        return (patient_hourly_vals ,(data_start_day),dci_dt_wrt_bd)# data not anchored

