import numpy as np
import pandas as pd
import statistics
from scipy.signal import savgol_filter
import scipy.stats as stats
import os
import scipy.signal as signal
Hz = 1
sampling_rate = 30 * Hz
second = 30
min = 60 * second
smooth_poly = 4
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other', 'Jump']
num_of_behaviors = len(behaviors)

# def find_bouts(data , behavior,bout_theta):
#     second = 15
#     bouts_len = []
#     IEI  = []
#     bout = []
#     last_bout_end = 0
#     first_flag = True
#     latency_to_first_bout=np.nan
#     for col in range(data.shape[0]):
#         if data[col] == behavior:
#             bout.append(col)
#         else:
#             if len(bout)>bout_theta:
#                 IEI.append(bout[0]-last_bout_end)
#                 bouts_len.append(len(bout))
#                 last_bout_end = bout[-1]
#                 if first_flag:
#                     latency_to_first_bout = bout[0]
#                     first_flag=False
#                 bout = []
#
#
#
#     return IEI,bouts_len,latency_to_first_bout


def is_last_seq_in_baseline_period (preds , b, minimal_bout_length):
    counter = 0
    for t in np.arange(preds.shape[0]-1,-1,-1):
        if preds[t]==b:
            counter+=1
        else:
            if counter>minimal_bout_length:
                return True
            else:
                counter=0
    if counter>minimal_bout_length:
        return True
    else:
        return False



def movemin (signal , window_size, aligment):
    new_signal = np.copy(signal)
    if aligment=='center':
        for i in range(len(signal)):
            window_start = np.max([0,i-window_size//2])
            window_end = np.min([len(signal),i+window_size//2])
            new_signal[i]=new_signal[i]-np.min(signal[window_start:window_end])
    else:
        for i in range(len(signal)-window_size):
            new_signal[i] = new_signal[i]-np.min(signal[i:i+window_size])
    return new_signal




def find_bouts_long_short_np_wp(data , behavior, short_long_threshold , threshold_for_bout_length=1):
    second = 15
    short_long_threshold=short_long_threshold*second
    window_end_to_bout_onset = 10 * second
    priors_period = int(3*second)
    np_bouts_len = []
    np_bouts  = {'short':[],'long':[]}
    wp_bouts_len = []
    wp_bouts = {'short':[],'long':[]}
    bout = []

    for col in range(priors_period,data.shape[0]-window_end_to_bout_onset):
        if data[col] == behavior:
            bout.append(col)
        else:
            if len(bout) > threshold_for_bout_length*second:
                if (not is_last_seq_in_baseline_period(preds=data[bout[0]-priors_period:bout[0]],b=behavior,minimal_bout_length=1*second)):
                    if len(bout) <= short_long_threshold:
                        np_bouts['short'].append(bout)
                    else:
                        np_bouts['long'].append(bout)
                else:
                    if len(bout) <= short_long_threshold:
                        wp_bouts['short'].append(bout)
                    else:
                        wp_bouts['long'].append(bout)

            bout = []

    return np_bouts,wp_bouts

def find_time_from_last_bout (preds , b, minimal_bout_length):
    counter = 0

    for t in np.arange(preds.shape[0]-1,-1,-1):
        if preds[t]==b:
            counter+=1
        else:
            if counter>minimal_bout_length:
                return preds.shape[0]-(t+counter)
            else:
                counter=0
    if counter>minimal_bout_length:
        return preds.shape[0]-counter
    else:
        return False


def find_distances_from_last_bout(data , behavior, threshold_for_bout_length=1):
    second = 15

    window_end_to_bout_onset = 10 * second
    priors_period = 20*second

    bouts  = {'bouts':[],'time_from_last_bout':[]}
    bout=[]
    for col in range(priors_period,data.shape[0]-window_end_to_bout_onset):
        if data[col] == behavior:
            bout.append(col)
        else:
            if len(bout) > threshold_for_bout_length*second :
                time_from_last_bout = find_time_from_last_bout(preds=data[bout[0]-priors_period:bout[0]],b=behavior,minimal_bout_length=1*second)
                if time_from_last_bout!=False:
                    bouts['bouts'].append(bout)
                    bouts['time_from_last_bout'].append(time_from_last_bout)
            bout = []

    return bouts

def get_signal(folder, file):

    data = pd.DataFrame(pd.read_csv(folder+file))
    iso_left = np.array(data.loc[data['Flags'] == 17,'Region3G'])
    iso_right = np.array(data.loc[data['Flags'] == 17, 'Region2G'])
    sig_left = np.array(data.loc[data['Flags'] == 18, 'Region3G'])
    sig_right = np.array(data.loc[data['Flags'] == 18, 'Region2G'])
#     print(iso_right.shape, iso_left.shape, sig_right.shape, sig_left.shape)

    #shortening arrays to the length of the either the iso or sig:
    number_of_samples = np.amin([iso_left.shape[0],sig_left.shape[0]])
    iso_left = iso_left[:number_of_samples]
    iso_right = iso_right[:number_of_samples]
    sig_left = sig_left[:number_of_samples]
    sig_right = sig_right[:number_of_samples]
    #     iso_left = iso_left[remove*second:number_of_samples]
    #     iso_right = iso_right[remove*second:number_of_samples]
    #     sig_left = sig_left[remove*second:number_of_samples]
    #     sig_right = sig_right[remove*second:number_of_samples]


    iso_left_coef = np.polyfit(iso_left, sig_left, deg =1)
    iso_right_coef = np.polyfit(iso_right, sig_right, deg =1)
    iso_left_fitted = np.polyval(iso_left_coef,iso_left)
    iso_right_fitted = np.polyval(iso_right_coef,iso_right)

    #smoothing the iso and signal:
    iso_left_smooth = savgol_filter(iso_left_fitted,window_length = 91, polyorder = smooth_poly)
    iso_right_smooth = savgol_filter(iso_right_fitted,window_length = 91, polyorder = smooth_poly)
    sig_left_smooth = savgol_filter(sig_left,window_length = 41, polyorder = smooth_poly+1)
    sig_right_smooth = savgol_filter(sig_right,window_length = 41, polyorder = smooth_poly+1)

    #calculating df/f:
    df_over_f_left = (sig_left_smooth - iso_left_smooth)/iso_left_smooth
    df_over_f_right = (sig_right_smooth - iso_right_smooth)/iso_right_smooth

    df_over_f_left = movemin(df_over_f_left, 2*min,'center')
    df_over_f_right = movemin(df_over_f_right, 2 * min, 'center')

    #transition to zscore:
    z_df_over_f_left = stats.zscore(df_over_f_left)
    z_df_over_f_right = stats.zscore(df_over_f_right)

    return z_df_over_f_left, z_df_over_f_right

def is_last_seq_in_post_baseline_period (preds , b, minimal_bout_length):
    counter = 0
    for t in np.arange(preds.shape[0]):
        if preds[t]==b:
            counter+=1
        else:
            if counter>minimal_bout_length:
                return True
            else:
                counter=0
    if counter>minimal_bout_length:
        return True
    else:
        return False

def find_bouts_with_post_quiet_period(data , behavior, short_long_threshold , threshold_for_bout_length=1):
    second = 15
    short_long_threshold=short_long_threshold*second
    quiet_period = int(5*second)
    np_bouts  = {'short':[],'long':[]}
    bout = []

    for col in range(0,data.shape[0]-quiet_period):
        if data[col] == behavior:
            bout.append(col)
        else:
            if len(bout) > threshold_for_bout_length*second:
                if (not is_last_seq_in_post_baseline_period(preds=data[bout[-1]:quiet_period+bout[-1]],b=behavior,minimal_bout_length=1*second) and
                        (not is_last_seq_in_baseline_period(preds=data[bout[0]-quiet_period:bout[0]],b=behavior,minimal_bout_length=1*second))):
                    if len(bout) <= short_long_threshold:
                        np_bouts['short'].append(bout)
                    else:
                        np_bouts['long'].append(bout)

            bout = []

    return np_bouts

def get_session_dynamics(session , behavior , num_of_bins):
    bins = np.linspace(0,session.shape[0],num_of_bins+1,dtype = np.int16)
    dynamics = []

    for i in range(num_of_bins):
        dynamics.append(np.count_nonzero(session[bins[i]:bins[i+1]]==behavior)/(bins[i+1]-bins[i]))
    return dynamics

def get_session_cumsum(session , behavior,num_of_bins ):
    bins = np.linspace(0,session.shape[0],num_of_bins+1,dtype = np.int16)
    session = np.where(session==behavior,1,0)
    cumsum = []
    for bin in bins:
        cumsum.append(np.count_nonzero(session[:bin]))
    return np.array(cumsum)/np.count_nonzero(session)
    #
    # for i in range(num_of_bins):
    #     dynamics.append(np.count_nonzero(session[bins[i]:bins[i+1]]==behavior)/(bins[i+1]-bins[i]))
    # return dynamics



def find_mouse(file):
    folder = 'Data/'
    for cohort in os.listdir(folder):
        if os.path.isdir(folder+cohort):
            for mouse in os.listdir(folder+cohort):
                # print(os.listdir(folder+cohort),file.replace('.pkl','_ethogram.png'))
                if os.path.exists(folder+cohort+'/'+mouse+'/'+file.replace('.pkl','_ethogram.png')):
                    return mouse

def smoothing(sig , window_len):
    smooth_window = np.ones(window_len) / window_len
    smoothed_sig = signal.filtfilt(smooth_window, 1, sig)
    return smoothed_sig


def segment_bouts(data,behavior,bout_theta):
    bouts_data = {'indices':[],'number':0,'length':[],'IBI':[]}
    behavior_indices = np.nonzero(data==behavior)[0]
    bout = []
    for i in range(behavior_indices.size-1):
        if behavior_indices[i+1]-behavior_indices[i]==1:
            bout.append(behavior_indices[i+1])
        else:
            if len(bout)>bout_theta:
                if len(bouts_data['IBI'])>0:
                    bouts_data['IBI'].append(bout[0] - bouts_data['indices'][-1][-1])
                else:
                    bouts_data['IBI'].append(bout[0])
                bouts_data['indices'].append(bout)
                bouts_data['length'].append(len(bout))
                bouts_data['number']+=1
            bout = []
    return bouts_data


def find_quantile(arr,q):
    arr  = np.sort(arr)
    i_percentile_q = (arr.size + 1) * q
    # print(np.arange(arr.size, dtype='double'),i_percentile_q)
    if i_percentile_q not in np.arange(arr.size,dtype='double'):
        rounded_i_per_q = int(np.floor(i_percentile_q))
        percentile_q = ((arr[rounded_i_per_q - 1] * 0.5) + (arr[rounded_i_per_q] * 0.5))
    else:
        percentile_q = arr[int(i_percentile_q)]
    return percentile_q

def segement_bouts_transition(arr, bout_theta):
    pi = {}
    for b_cur in range(num_of_behaviors):
        pi[b_cur]={}
        for b_next in range(num_of_behaviors):
            pi[b_cur][b_next]=[]
    bout = []
    for i in range(arr.size - 1):
        if arr[i] == arr[i + 1]:
            bout.append(arr[i])
        else:
            if len(bout) > bout_theta:
                pi[arr[i]][arr[i+1]].append(i)
            bout = []
    return pi
def find_Q(arr,q):
    arr = np.sort(arr)
    i_percetile_q = (arr.size+1)*q
    if i_percetile_q not in np.arange(arr.size,dtype='double'):
        rounded_percetile_q = int(np.floor((i_percetile_q)))
        percentile_q = arr[rounded_percetile_q-1]*0.5+arr[rounded_percetile_q]*0.5
    else:
        percentile_q = arr[int(i_percetile_q)]
    return percentile_q

def remove_IQR_outliers(diff_arr):
    diff_arr = np.array(diff_arr)
    Q1 = find_Q(diff_arr, 0.25)
    Q3 = find_Q(diff_arr, 0.75)
    IQR = Q3 - Q1
    low_cutoff = Q1 - (1.5 * IQR)
    high_cutoff = Q3 + (1.5 * IQR)
    diff_arr[diff_arr < low_cutoff] = np.nan
    diff_arr[diff_arr > high_cutoff] = np.nan
    return diff_arr



def groupedByDuration_bouts(data , behavior, segments ):
    second = 15
    minimal_bout_length = 1*second
    window_end_to_bout_onset = 10 * second
    priors_period = int(3*second)
    bouts = {}
    for seg in segments:
        bouts[seg] = []
    bout = []
    for col in range(priors_period,data.shape[0]-window_end_to_bout_onset):
        if data[col] == behavior:
            bout.append(col)
        else:
            # print(len(bout))
            if len(bout) > minimal_bout_length:
                if (not is_last_seq_in_baseline_period(preds=data[bout[0]-priors_period:bout[0]],b=behavior,minimal_bout_length=1*second)):
                    for t in range(len(segments)-1):
                        if len(bout)>=segments[t]*second and len(bout)<segments[t+1]*second:
                            bouts[segments[t]].append(bout)
                            # print(len(bout),'added to segment',segments[t])
                    if len(bout)>segments[-1]*second:
                        bouts[segments[-1]].append(bout)
                        # print(len(bout), 'added to segment', segments[-1])
            bout = []


    return bouts


def quantify_transitions(data,B):
    second = 15
    bout_segmented_data = np.diff(data,axis=0)
    transitions_points = np.nonzero(bout_segmented_data)[0]
    transitions = np.zeros((B,B))
    bouts = []
    bouts.append([data[0], transitions_points[0]+1])
    last_tp = transitions_points[0]
    for tp in transitions_points[1:]:
        bouts.append([data[tp],tp-last_tp])
        last_tp = tp
    bouts.append([data[-1],data.size-last_tp])
    bouts = np.array(bouts)
    for behavior in range(B):
        for idx in range(1,bouts.shape[0],1):
            if bouts[idx,0]==behavior and bouts[idx,1]>=1*second :
                if bouts[idx-1,1]>0.5*second:
                    transitions[bouts[idx-1,0],behavior]+=1
    return transitions

def get_PTT(data,B,photom):
    second = 45
    pre_window = 45
    post_window = second
    bout_segmented_data = np.diff(data,axis=0)
    transitions_points = np.nonzero(bout_segmented_data)[0]
    PTT = {}
    for b1 in range(B):
        for b2 in range(B):
            PTT[(b1,b2)] = []
    bouts = []
    bouts.append([data[0], transitions_points[0]+1,transitions_points[0]+1])
    last_tp = transitions_points[0]
    for tp in transitions_points[1:]:
        bouts.append([data[tp],tp-last_tp,tp+1])
        last_tp = tp

    bouts.append([data[-1],data.size-last_tp,None])
    bouts = np.array(bouts)
    for behavior in range(B):
        for idx in range(1,bouts.shape[0],1):
            if bouts[idx,0]==behavior and bouts[idx,1]>=1*second and bouts[idx-1,1]>1*second :
                tp = bouts[idx-1,2]
                if tp - pre_window > 0 and tp+pre_window< data.size:
                    if len(photom)==1:
                        PTT[(bouts[idx-1,0],behavior)].append(photom[0][tp-pre_window:tp+post_window])
                    else:
                        PTT[(bouts[idx - 1, 0], behavior)].append(photom[0][tp - pre_window:tp + post_window])
                        PTT[(bouts[idx - 1, 0], behavior)].append(photom[1][tp - pre_window:tp + post_window])

    return PTT

def find_bouts(data , behavior, pre_bout_quiet_period = 3, threshold_for_bout_length=1):
    second = 15
    window_end_to_bout_onset = 10 * second
    priors_period = int(pre_bout_quiet_period*second)
    bouts  = []
    bout = []

    for col in range(priors_period,data.shape[0]-window_end_to_bout_onset):
        if data[col] == behavior:
            bout.append(col)
        else:
            if len(bout) > threshold_for_bout_length*second:
                if (not is_last_seq_in_baseline_period(preds=data[bout[0]-priors_period:bout[0]],b=behavior,minimal_bout_length=1*second)):
                    bouts.append(bout)
            bout = []

    return bouts
