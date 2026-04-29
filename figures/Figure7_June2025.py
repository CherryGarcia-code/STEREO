#%% Imports
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import copy
import os

import numpy as np
import bz2
import scipy.signal
import seaborn as sns
from matplotlib import pyplot as plt

import helper_functions as hf
import scipy.signal as signal
import pickle

import scipy.stats as stats
import sys

root_folder = '.'
folder = root_folder+'Data/'
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
FPS=15
second = 15
minute = 60*second
NO_LICKING = 0
SELF_LICKING = 1
SURFACE_LICKING = 2
UNDEFINED=10
FLOOR_LICKING = 20
WALL_LICKING = 30
GROOMING = 40
BODY_LICKING = 50
REARING = 60
LOCOMOTION = 70
STATIONARY = 80
behaviors_labels = {NO_LICKING:'No licking',
                 SELF_LICKING:'Self licking',
                 SURFACE_LICKING:'Surface licking',
                 FLOOR_LICKING:'Floor licking',
                 WALL_LICKING:'Wall licking',
                 GROOMING:'Grooming',
                 BODY_LICKING:'Body licking',
                 REARING:'Rearing',
                 LOCOMOTION:'Locomotion',
                 STATIONARY:'Stationary'}
ifile = bz2.BZ2File(folder + 'CTM_Dec24.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
trials = ['saline1', 'saline2', 'saline3', 'cocaine1','cocaine2', 'cocaine3','cocaine4', 'cocaine5','splashTest']
grouped_trials = ['saline','cocaine1','cocaine2', 'cocaine3','cocaine4', 'cocaine5','splashTest']
#Remove corrupted signals due to bad infection/fiber location
for tr in trials:
    # Remove mice
    try:del CMT['a2a_hm3dq']['cA242m8'][tr]['photom']
    except KeyError:print('No trial ', tr,' for mouse cA242m8')
    try:del CMT['drd1_hm3dq']['c514Bm2'][tr]['photom']
    except KeyError: print('No trial ', tr,' for mouse c514Bm2')
    try:del CMT['drd1_hm3dq']['c514Bm8'][tr]['photom']
    except KeyError:print('No trial ', tr,' for mouse c514Bm8')
    #Remove hemispheres
    try:del CMT['controls']['c548m8'][tr]['photom']['left']
    except  KeyError: print('No trial ', tr,' for mouse c548m8')
    try:del CMT['controls']['c548m11'][tr]['photom']['left']
    except KeyError: print('No trial ', tr,' for mouse c548m11')
    try:del CMT['a2a_hm3dq']['cA242m5'][tr]['photom']['left']
    except KeyError:print('No trial ', tr,' for mouse cA242m5')
#Remove additional corrupted trials
del CMT['drd1_hm4di']['c528m10']['cocaine4']['photom']
del CMT['controls']['c548m11']['cocaine5']['photom']
del CMT['a2a_hm3dq']['cA242m6']['saline2']['photom']
del CMT['a2a_hm3dq']['cA242m6']['cocaine5']['photom']
del CMT['a2a_hm3dq']['cA184m4']['saline3']['photom']
del CMT['controls']['cA242m9']['saline2']['photom']['left']
del CMT['controls']['cA242m9']['cocaine1']['photom']['right']
del CMT['a2a_hm3dq']['cA184m7']['cocaine4']['photom']['left']
# Remove manually identified corrupted segments within good sessions
CMT['controls']['cA242m4']['saline1']['photom']['left']['df'][3500:3800] = np.nan
CMT['controls']['cA242m4']['saline1']['photom']['left']['z'] = stats.zscore(CMT['controls']['cA242m4']['saline1']['photom']['left']['df'],nan_policy='omit')
CMT['controls']['cA242m4']['cocaine1']['photom']['left']['df'][23500:24000] = np.nan
CMT['controls']['cA242m4']['cocaine1']['photom']['left']['z'] = stats.zscore(CMT['controls']['cA242m4']['cocaine1']['photom']['left']['df'],nan_policy='omit')
CMT['controls']['cA242m9']['cocaine1']['photom']['left']['df'][9400:9800] = np.nan
CMT['controls']['cA242m9']['cocaine1']['photom']['left']['z'] = stats.zscore(CMT['controls']['cA242m9']['cocaine1']['photom']['left']['df'],nan_policy='omit')
CMT['drd1_hm4di']['c528m5']['cocaine5']['photom']['right']['df'][16600:16800] = np.nan
CMT['drd1_hm4di']['c528m5']['cocaine5']['photom']['right']['z'] = stats.zscore(CMT['drd1_hm4di']['c528m5']['cocaine5']['photom']['right']['df'],nan_policy='omit')
for seg in [np.arange(700,1500),np.arange(6200,7000),np.arange(11300,11700),np.arange(13400,14400),np.arange(15700,15720),np.arange(22800,25000)]:
    CMT['controls']['cA242m4']['cocaine2']['photom']['left']['df'][seg] = np.nan
CMT['controls']['cA242m4']['cocaine2']['photom']['left']['z'] = stats.zscore(CMT['controls']['cA242m4']['cocaine2']['photom']['left']['df'],nan_policy='omit')
print('***Corrupted signal were removed***')
selfLicking_lut =  np.array([NO_LICKING,NO_LICKING,SURFACE_LICKING,SURFACE_LICKING,GROOMING,BODY_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
noLicking_lut =  np.array([UNDEFINED,UNDEFINED,FLOOR_LICKING,WALL_LICKING,SELF_LICKING,SELF_LICKING,REARING,LOCOMOTION,STATIONARY])
goFrom_behavior = NO_LICKING
goTo_behavior = BODY_LICKING
goFrom_label = behaviors_labels[goFrom_behavior]
goTo_label = behaviors_labels[goTo_behavior]
boi_lut = selfLicking_lut
mm=1/25.4
#%% Fig 7E
output_folder = 'output/Photometry/forPaper/behavior_overlaid_on_signal/'
sig_type='z'
smooth_window=3
tr = ['splashTest']
# c = 'controls'
c = 'a2a_hm3dq'
# m = 'c548m10'
m= 'cA242m6'
tr = 'splashTest'
# window = np.arange(8*minute,12*minute,1)
window = np.arange(7*minute,11*minute,1)
photom_offset = CMT[c][m][tr]['photom']['offset']
predictions = np.copy(CMT[c][m][tr]['merged']['predictions']['smartMerge'])[window]
plt.figure(figsize=(75 * mm, 30 * mm))
left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)[window]
right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)[window]

plt.vlines(x=np.argwhere(predictions==5)[:,0],ymin=-2,ymax=7,linewidth=1.0/15.0,colors=colors[5])
plt.plot(left, label='Left hemisphere', c='green',lw=.5)
# plt.plot(right, label='Right hemisphere', c='green',lw=.5)


plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# plt.ylabel('5% $\Delta{F}/{F}$',fontsize=8)
# plt.legend(frameon=False,fontsize=8)
plt.xticks([1*minute],[''])
plt.savefig(output_folder+c+'_'+m+'_'+tr+'.png',bbox_inches='tight',dpi=300)
plt.savefig(output_folder+c+'_'+m+'_'+tr+'.pdf',bbox_inches='tight', dpi=300)
plt.close()
#%% Fig 7F
goFrom_dur = 5
smooth_signal = True
smooth_window = 3
output_folder = 'output/Photometry/forPaper/'
photometry = {}
bout_length = {}
events_pre_mouse = {}
seconds = 15
bout_onset = 5*seconds
mm=1/25.4
sig_type = 'z'
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    events_pre_mouse[pathway] = {}
    for tr in grouped_trials:
        photometry[pathway][tr]= []
        bout_length[pathway][tr]= []
        events_pre_mouse[pathway][tr] = {}

pre_switch_window = 5*seconds
post_switch_window = 5*seconds
baseline_window_end = 2*seconds
window_length = pre_switch_window+post_switch_window

for c in CMT:
    for m in CMT[c]:
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = 'drd1'
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = 'a2a'
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = 'a2a'
            else:
                pathway = 'drd1'
        else:
            break
        for tr in CMT[c][m]:
            if tr not in trials or 'photom' not in CMT[c][m][tr] : continue

            if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                side = 'both'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
                    right = CMT[c][m][tr]['photom']['right'][sig_type]
            elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                side = 'left'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
            else:
                side = 'right'
                if smooth_signal:
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    right = CMT[c][m][tr]['photom']['right'][sig_type]

            predictions = boi_lut[CMT[c][m][tr]['merged']['predictions']['smartMerge']]
            bouts = hf.segment_bouts_per_history(predictions, goTo_behavior,goFrom_behavior,min_duration_for_hisotry=goFrom_dur)

            for bout_idx in range(len(bouts)):
                if 'saline' in tr: tr = 'saline'
                if side== 'left':
                    photom_seg = hf.segment_signal_window(left,bouts[bout_idx][0],'df',pre_switch_window,post_switch_window)
                    if not np.all(np.isnan(photom_seg)):
                        bout_length[pathway][tr].append(len(bouts[bout_idx]))
                        photometry[pathway][tr].append(photom_seg)

                elif side == 'right':
                    photom_seg =  hf.segment_signal_window(right,bouts[bout_idx][0],'df',pre_switch_window,post_switch_window)
                    if not np.all(np.isnan(photom_seg)):
                        bout_length[pathway][tr].append(len(bouts[bout_idx]))
                        photometry[pathway][tr].append(photom_seg)
                else:
                    photom_seg_left = hf.segment_signal_window(left, bouts[bout_idx][0], 'df',pre_switch_window, post_switch_window)
                    photom_seg_right = hf.segment_signal_window(right, bouts[bout_idx][0], 'df',pre_switch_window, post_switch_window)
                    if not np.all(np.isnan(photom_seg_left)):
                        bout_length[pathway][tr].append(len(bouts[bout_idx]))
                        photometry[pathway][tr].append(photom_seg_left)
                    if not np.all(np.isnan(photom_seg_right)):
                        bout_length[pathway][tr].append(len(bouts[bout_idx]))
                        photometry[pathway][tr].append(photom_seg_right)

cohorts = ['drd1','a2a']
trials_subset = [grouped_trials[0],grouped_trials[6]]
c_dspns = '#8e63b8'
c_ispns = '#3d1f4d'
lims = [0, 2]
num_of_trials = len(trials_subset)
fig,ax = plt.subplots(ncols=num_of_trials ,nrows =3,sharex='all', gridspec_kw={'height_ratios':[3,1,1]},figsize=(65*mm,55*mm))
tr_idx=0
for tr in trials_subset:
    row_idx = 0
    dspns = np.array(photometry['drd1'][tr],dtype=np.float32)
    ispns = np.array(photometry['a2a'][tr],dtype=np.float32)
    dspns_dur = np.array(bout_length['drd1'][tr])
    sorted_indices = np.argsort(dspns_dur)
    dspns = dspns[sorted_indices, :]
    ispns_dur = np.array(bout_length['a2a'][tr])
    sorted_indices = np.argsort(ispns_dur)
    ispns = ispns[sorted_indices, :]
    mean_dspns = np.nanmean(dspns,axis=0)
    err_dspns = np.sqrt(np.nanvar(dspns,axis=0)/dspns.shape[0])
    mean_ispns = np.nanmean(ispns,axis=0)
    err_ispns = np.sqrt(np.nanvar(ispns,axis=0)/ispns.shape[0])

    plt.sca(ax[0, tr_idx])
    plt.plot(mean_dspns,c = c_dspns,label = 'dSPN',lw=.7)
    plt.fill_between(np.arange(mean_dspns.size),y1=mean_dspns-err_dspns,y2=mean_dspns+err_dspns,color=c_dspns,alpha=.3)
    plt.plot(mean_ispns, c=c_ispns, label = 'iSPN',lw=.7)
    plt.fill_between(np.arange(mean_ispns.size), y1=mean_ispns - err_ispns, y2=mean_ispns + err_ispns, color=c_ispns, alpha=.3)
    plt.vlines(x=pre_switch_window, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='--', colors='k',lw=.5)
    if tr_idx==1:
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tick_params(axis='y', which='both', left=False, labelleft=False)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(frameon=False,fontsize=8)
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.ylim(lims)

    if dspns.shape[0]>0:
        plt.sca(ax[1, tr_idx])
        sns.heatmap(dspns, ax=ax[1, tr_idx],vmin=-.5,vmax=1.5, cbar=False,cmap='magma')
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    if ispns.shape[0]>0:
        plt.sca(ax[2, tr_idx])
        sns.heatmap(ispns,ax=ax[2, tr_idx],vmin=-.5,vmax=1.5, cbar=False,cmap='magma')

    if tr_idx == 0:
        ax[0, 0].set_ylabel('Mean $\Delta$F/F(Z)',rotation=90,va='bottom',ha='center',fontsize=10)
        ax[1, 0].set_ylabel('dSPN',rotation=90,va='bottom',ha='center',fontsize=10)
        ax[2, 0].set_ylabel('iSPN',rotation=90,va='bottom',ha='center',fontsize=10)

    ax[1, tr_idx].set_yticks([dspns.shape[0]],[dspns.shape[0]],fontsize=8,va='bottom')
    ax[1, tr_idx].tick_params(axis='y',length=2,width=.5,pad=.5)
    ax[2, tr_idx].set_yticks([ispns.shape[0]],[ispns.shape[0]],fontsize=8,va='bottom')
    ax[2, tr_idx].tick_params(axis='y',length=2,width=.5,pad=.5)
    plt.sca(ax[2,tr_idx])
    plt.xticks(np.arange(1 * seconds, window_length+seconds, 2 * seconds), (np.arange(-(pre_switch_window-seconds), (window_length-pre_switch_window), 2*seconds)/seconds).astype(np.int16), rotation=0,fontsize=8)
    tr_idx+=1
ax[0,0].set_yticks([0,.75,1.5],[0,.75,1.5], fontsize=8)
ax[0, 0].set_ylim([-.5, 1.5])
ax[0, 1].set_ylim([-.5, 1.5])
os.makedirs(output_folder+'/Summary/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_'),exist_ok=True)
plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0.5)
plt.savefig(output_folder+'/Summary/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_')+'/raw_ERTs.png',dpi=300)
plt.savefig(output_folder+'/Summary/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_')+'/raw_ERTs.pdf',dpi=300)
plt.close()
#%% Fig 7G
seconds=15
smooth_signal=True
smooth_window=3
SAL=grouped_trials[0]
SPLASH=grouped_trials[6]
trials_subset = [SAL,SPLASH]
colors_dur = ['#3B82F6', '#6479B3', '#6A5A99', '#8E63B8']
ticks = [0,2]
num_of_trials = len(trials_subset)
output_folder = 'output/Photometry/forPaper/durationBased_quartiles_separation_SRTs'
pre_switch_window = 3 * seconds
post_switch_window = 5 * seconds
baseline_period = 2*seconds
switch_window= pre_switch_window+post_switch_window
goFrom_dur=5
sig_type='z'
SRTs = {'dSPN':{},'iSPN':{}}
for tr in trials_subset:
    SRTs['dSPN'][tr] = {'traces':[],'bout_durations':[]}
    SRTs['iSPN'][tr] = {'traces': [], 'bout_durations': []}
for c in ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq']:
    for m in CMT[c]:
        if ('a2a' in c) or (c=='controls' and m.startswith('cA')):
            SPN ='iSPN'
        else:
            SPN = 'dSPN'
        for tr in trials:
            if tr not in CMT[c][m] or 'photom' not in CMT[c][m][tr] : continue
            if 'saline' not in tr and tr not in trials_subset: continue
            if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom']:
                side = 'both'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
                    right = CMT[c][m][tr]['photom']['right'][sig_type]
            elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                side = 'left'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
            else:
                side = 'right'
                if smooth_signal:
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    right = CMT[c][m][tr]['photom']['right'][sig_type]
            predictions = boi_lut[CMT[c][m][tr]['merged']['predictions']['smartMerge']]
            bouts = hf.segment_bouts_per_history(predictions, goTo_behavior, goFrom_behavior,
                                             min_duration_for_hisotry=goFrom_dur)
            for bout_idx in range(len(bouts)):
                if 'saline' in tr: tr = 'saline'
                bout_dur = len(bouts[bout_idx])
                align_idx=bouts[bout_idx][0]
                if side == 'left':
                    photom_seg = hf.segment_signal_window(left, align_idx, 'f', pre_switch_window, post_switch_window)
                    if not np.all(np.isnan(photom_seg)):
                       if goTo_behavior in [GROOMING, BODY_LICKING]:photom_seg = photom_seg-np.nanmean(photom_seg[:baseline_period]) # turn on
                       SRTs[SPN][tr]['traces'].append(photom_seg)
                       SRTs[SPN][tr]['bout_durations'].append(bout_dur)
                elif side == 'right':
                    photom_seg = hf.segment_signal_window(right, align_idx, 'f', pre_switch_window, post_switch_window)
                    if not np.all(np.isnan(photom_seg)):
                        if goTo_behavior in [GROOMING, BODY_LICKING]:photom_seg = photom_seg - np.nanmean(photom_seg[:baseline_period])
                        SRTs[SPN][tr]['traces'].append(photom_seg)
                        SRTs[SPN][tr]['bout_durations'].append(bout_dur)
                else:
                    photom_seg_left = hf.segment_signal_window(left, align_idx, 'df', pre_switch_window,
                                                               post_switch_window)
                    photom_seg_right = hf.segment_signal_window(right, align_idx, 'df', pre_switch_window,
                                                                post_switch_window)
                    if not np.all(np.isnan(photom_seg_left)):
                        if goTo_behavior in [GROOMING, BODY_LICKING]:photom_seg_left = photom_seg_left - np.nanmean(photom_seg_left[:baseline_period])
                        SRTs[SPN][tr]['traces'].append(photom_seg_left)
                        SRTs[SPN][tr]['bout_durations'].append(bout_dur)
                    if not np.all(np.isnan(photom_seg_right)):
                        if goTo_behavior in [GROOMING, BODY_LICKING]:photom_seg_right = photom_seg_right - np.nanmean(photom_seg_right[:baseline_period])
                        SRTs[SPN][tr]['traces'].append(photom_seg_left)
                        SRTs[SPN][tr]['bout_durations'].append(bout_dur)

trials_to_present = [trials_subset[-1]]
fig,ax = plt.subplots(nrows=2,frameon=False,figsize = (30 * mm, 50 * mm),sharex='all')
tr_idx=0
for tr in trials_to_present:
    print(tr)
    SPN_idx=0
    for SPN in ['dSPN','iSPN']:
        plt.sca(ax[SPN_idx])
        traces = np.copy(SRTs[SPN][tr]['traces'])
        durations = np.copy(SRTs[SPN][tr]['bout_durations'])
        q1 = hf.find_quantile(durations,.25)
        q2 = hf.find_quantile(durations, .5)
        q3 = hf.find_quantile(durations, .75)
        q1_indices = np.argwhere(durations<q1)[:,0]
        q2_indices = np.argwhere((durations > q1) & (durations<q2))[:,0]
        q3_indices = np.argwhere((durations > q2) & (durations<q3))[:,0]
        q4_indices = np.argwhere(durations > q3)[:,0]

        q1_mean = np.nanmean(traces[q1_indices,:],axis=0)
        q1_err = np.sqrt(np.nanvar(traces[q1_indices, :], axis=0)/q1_indices.size)
        q2_mean = np.nanmean(traces[q2_indices, :], axis=0)
        q2_err = np.sqrt(np.nanvar(traces[q2_indices, :], axis=0) / q2_indices.size)
        q3_mean = np.nanmean(traces[q3_indices, :], axis=0)
        q3_err = np.sqrt(np.nanvar(traces[q3_indices, :], axis=0) / q3_indices.size)
        q4_mean = np.nanmean(traces[q4_indices, :], axis=0)
        q4_err = np.sqrt(np.nanvar(traces[q4_indices, :], axis=0) / q4_indices.size)

        plt.plot(q1_mean,label='Q1',color=colors_dur[0],lw=.7)
        plt.fill_between(x= np.arange(switch_window),y1=q1_mean-q1_err,y2=q1_mean+q1_err,alpha=.3,color=colors_dur[0])
        plt.plot(q2_mean, label='Q2', color=colors_dur[1],lw=.7)
        plt.fill_between(x=np.arange(switch_window), y1=q2_mean - q1_err, y2=q2_mean + q2_err, alpha=.3,
                         color=colors_dur[1])
        plt.plot(q3_mean, label='Q3', color=colors_dur[2],lw=.7)
        plt.fill_between(x=np.arange(switch_window), y1=q3_mean - q3_err, y2=q3_mean + q3_err, alpha=.3,
                         color=colors_dur[2])
        plt.plot(q4_mean, label='Q4', color=colors_dur[3],lw=.7)
        plt.fill_between(x=np.arange(switch_window), y1=q4_mean - q4_err, y2=q4_mean + q4_err, alpha=.3,
                         color=colors_dur[3])
        plt.vlines(x=pre_switch_window,ymin = plt.gca().get_ylim()[0],ymax = plt.gca().get_ylim()[1],lw=.5,ls='--',color='k')
        plt.yticks(ticks,ticks,fontsize=8)
        if SPN_idx==0: plt.xticks([])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        SPN_idx+=1
    plt.legend(frameon=False, fontsize=8)
    plt.xticks(np.arange(1 * seconds, switch_window + seconds, 2 * seconds), (
            np.arange(-(pre_switch_window - seconds), (switch_window - pre_switch_window),
                      2 * seconds) / seconds).astype(np.int16), rotation=0, fontsize=8)
    tr_idx+=1
plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0.1)
os.makedirs(output_folder+'/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_')+'/'+trials_to_present[0],exist_ok=True)
plt.savefig(output_folder+'/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_')+'/'+trials_to_present[0]+'/onset.png',dpi=300)
plt.savefig(output_folder+'/'+goFrom_label.replace(' ','_')+'2'+goTo_label.replace(' ','_')+'/'+trials_to_present[0]+'/onset.pdf',dpi=300)
plt.close()
#%% Load Splash & opto data
sys.modules[__name__].__dict__.clear()
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CTM_Aug24.pkl', 'rb')
stim_days = pickle.load(ifile)['a2a_opto']
stim_days = {key:stim_days[key] for key in ['splashTest1p2mW','splashTest2p2mW']}
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
ISI = 1 * minute + 20*second
PRE = 0
DURING = 1
POST = 2
VDB = 0
nVDB = 1
bin_duration = 20 * second
num_of_bins = 3
num_of_behaviors = len(behaviors)
num_of_stims = 10
RNN_offset = 7
PATHO_LICKING = 0
NATURAL_LICKING = 1
NO_LICKING = 2
grouping_lut =  np.array([NO_LICKING,NO_LICKING,PATHO_LICKING,PATHO_LICKING,NATURAL_LICKING,NATURAL_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
#%% Fig 7J
output_folder = root_folder+'/Figures/Optogenetics/A2a_splashTest/Ethograms/'
first_stim = 3 * minute - RNN_offset
stim_times = np.arange(first_stim, 14 * minute, ISI)
for t in stim_days.keys():
    for mouse in stim_days[t]:
        print(mouse)
        predictions = stim_days[t][mouse]['merged']
        velocity = stim_days[t][mouse]['topcam']['velocity']
        SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
        stims = np.empty_like(predictions)
        stims[:] = 10
        for stim in stim_times:
            stims[stim:stim + 20 * second] = 9
        predictions = np.reshape(predictions, (1, predictions.shape[0]))
        stims = np.reshape(stims, (1, stims.shape[0]))
        fig, ax = plt.subplots(nrows=4, sharex='all', figsize=(4, 1.6), gridspec_kw={'height_ratios': [1, 4, 3, 3]})

        ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors, cbar=False, vmin=0, vmax=num_of_behaviors - 1,
                            ax=ax[1])
        ax[1].tick_params(left=False, bottom=False)
        plt.sca(ax[1])
        ax[0] = sns.heatmap(stims, yticklabels=[''], cbar=False, cmap=['#B2E2F6', 'white'], vmin=9, vmax=10, ax=ax[0])
        ax[0].tick_params(left=False, bottom=False)
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        plt.sca(ax[0])
        ax[2].plot(velocity * FPS, lw=.3)
        ax[2].bar(x=np.nonzero(stims == 9)[1], height=np.max(velocity) * FPS, width=1, color='#B2E2F6')
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        plt.sca(ax[2])
        plt.yticks([0, 0.1], [0, 0.1], fontsize=8)
        ax[3].plot(SSD, lw=.3)
        ax[3].bar(x=np.nonzero(stims == 9)[1], height=np.max(SSD[~np.isnan(SSD)]), width=1, color='#B2E2F6')
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        plt.sca(ax[3])
        plt.yticks([0, 0.005], [0, 5], fontsize=8)
        plt.xticks([0, 2700], [0, 3], rotation=0, fontsize=8)
        fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
        plt.savefig(output_folder + '/'+t+'_'+ mouse + '.pdf', dpi=300)
        plt.savefig(output_folder + '/' + t + '_' + mouse + '.png', dpi=300)
        plt.close()
#%% Fig 7K
output_folder = root_folder+'/Figures/Optogenetics/A2a_splashTest/Summary/'
first_stim = 3 * minute - RNN_offset
stim_times = np.arange(first_stim, 14 * minute, ISI)
laser_status = np.zeros(14 * minute)
smooth_window_len = 60
smooth_window = np.ones(smooth_window_len) / smooth_window_len

for t in ['splashTest1p2mW','splashTest2p2mW']:
    m_idx = 0
    for mouse in stim_days[t]:
        if mouse=='cA180m5':continue
        print(t,mouse,stim_days[t][mouse]['merged'].shape)
        m_predictions = np.reshape(np.copy(stim_days[t][mouse]['merged']),((1,stim_days[t][mouse]['merged'].size)))
        print(m_predictions.shape)
        if m_idx==0:
            predictions = np.copy(m_predictions)
        else:
            delta = predictions.shape[1]-m_predictions.shape[1]
            if delta>0:
                predictions=predictions[:,:-delta]
            elif delta<0:
                m_predictions = m_predictions[:,:delta]
            print(predictions.shape,m_predictions.shape)
            predictions = np.vstack([predictions,m_predictions])
        m_idx+=1
    print(predictions.shape)
    dist = np.zeros((num_of_behaviors,predictions.shape[1]))
    for idx in range(predictions.shape[1]):
        for b in range(num_of_behaviors):
            dist[b,idx]=np.count_nonzero(predictions[:,idx]==b)/predictions.shape[0]

    fig,ax = plt.subplots(nrows=2,sharex='all',gridspec_kw={'height_ratios': [1, 10]},figsize=(4,2))
    time = np.arange(predictions.shape[1])
    for row in range(dist.shape[0]):
        dist[row,:]= signal.filtfilt(smooth_window, 1, dist[row,:])
    plt.sca(ax[1])
    plt.stackplot(time,dist[0],dist[1],dist[2],dist[3],dist[4],dist[5],dist[6],dist[7],dist[8],labels = behaviors,colors=colors)
    plt.xticks(np.arange(0,predictions.shape[1],900),np.arange(16),fontsize=8)
    plt.xlabel('Time[m]',fontsize=10)
    plt.ylabel('Probability',fontsize=10)
    plt.legend(frameon=False,fontsize=8)
    plt.xlim(0,predictions.shape[1])
    plt.ylim(0,1)
    plt.yticks([0,0.5,1],[0,0.5,1],fontsize=8)
    plt.gca().tick_params(left=False, bottom=False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.sca(ax[0])
    plt.bar(stim_times,height=1,width=bin_duration,color='#B2E2F7',align='edge')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().tick_params(left=False, bottom=False,labelleft=False)
    fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder+'stackplot'+t+'.png',dpi=300)
    plt.savefig(output_folder + 'stackplot' + t + '.pdf', dpi=300)
    plt.close()
#%% Fig 7L
output_folder = root_folder+'/Figures/Optogenetics/A2a_splashTest/Summary/'
first_stim = 3 * minute - RNN_offset
stim_times = np.arange(first_stim, 14 * minute, ISI)
for t in ['splashTest1p2mW','splashTest2p2mW']:
    m_idx = 0
    num_of_mice = len(stim_days[t].keys())
    stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims, num_of_mice, num_of_behaviors), dtype=np.float32)
    outliers = []
    for mouse in stim_days[t]:
        print(mouse)
        predictions = stim_days[t][mouse]['merged']
        velocity = stim_days[t][mouse]['topcam']['velocity']
        SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
        for stim_idx in range(stim_times.size):
            laser_onset = stim_times[stim_idx]
            for behavior in range(num_of_behaviors):
                stim_epoch_behavior[PRE, stim_idx, m_idx, behavior]= np.count_nonzero(predictions[laser_onset-bin_duration:laser_onset]==behavior)/bin_duration
                stim_epoch_behavior[DURING, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset: laser_onset + bin_duration] == behavior)/bin_duration
                stim_epoch_behavior[POST, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset + bin_duration:laser_onset+ (2*bin_duration)] == behavior)/bin_duration
        m_idx+=1
    num_of_mice_wo_outliers = num_of_mice-len(outliers)
    stim_epoch_behavior = stim_epoch_behavior[:,:,:num_of_mice_wo_outliers,:]
    for behavior in range(num_of_behaviors):
        plt.figure(figsize=(1.35, 1.35), frameon=False)
        plt.plot(np.mean(stim_epoch_behavior[:, :, :, behavior], axis=1), alpha=0.3, color='gray', marker='o',
                 markersize=1, lw=.5)
        plt.bar(1, height=1, color='#009FE3', alpha=.3)
        plt.errorbar(np.arange(3), y=np.mean(stim_epoch_behavior[:, :, :, behavior], axis=(1, 2)),
                     yerr=np.sqrt(np.var(stim_epoch_behavior[:, :, :, behavior]) / (
                                 stim_epoch_behavior.shape[1] * stim_epoch_behavior.shape[2])).T,
                     color=colors[behavior], capsize=2, capthick=.5, lw=.7)
        plt.xticks(np.arange(3), ['Prior', 'During', 'Post'], fontsize=8)
        plt.ylabel('% Time spent', fontsize=10)
        plt.yticks([0, 0.25,0.5], [0,25,50], fontsize=8)
        plt.ylim([0, .5])
        axes = plt.gca()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
        plt.savefig(output_folder + '/V_graph_'+t+'_' + behaviors[behavior] + '.png', dpi=300)
        plt.savefig(output_folder + '/V_graph_'+t+'_' + behaviors[behavior] + '.pdf', dpi=300)
        plt.close()
print('Outliers : ',outliers)
#%% Fig 7M
output_folder = root_folder+ '/Figures/Optogenetics/A2a_splashTest/Dynamics/'
for t in ['splashTest1p2mW','splashTest2p2mW']:
    num_of_mice = len(stim_days[t].keys())
    dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
    m_idx = 0
    first_stim = 3 * minute - RNN_offset
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    outliers = []
    for mouse in stim_days[t]:
        print(mouse)
        predictions = stim_days[t][mouse]['merged']
        velocity = stim_days[t][mouse]['topcam']['locomotion']
        SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
        for stim_idx in range(stim_times.size):
            laser_onset = stim_times[stim_idx]
            for behavior in range(num_of_behaviors):
                dynamics[m_idx , stim_idx, behavior,:] = predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==behavior
        m_idx+=1
    num_of_mice_wo_outliers = num_of_mice-len(outliers)
    dynamics = dynamics[:num_of_mice_wo_outliers,:,:,:]
    for behavior in range(num_of_behaviors):
        psth = np.mean(dynamics[:, :, behavior, :], axis=1)
        for m in range(num_of_mice):
            psth[m] = hf.smoothing(psth[m], 5)
        mean_psth = np.mean(psth, axis=0)
        stderr = np.sqrt(np.var(psth, axis=0) / psth.shape[0])
        plt.figure(figsize=(1.35, 1.5), frameon=False)
        plt.bar(x=bin_duration, height=np.ones(bin_duration), width=bin_duration, align='edge', color='#B2E2F7')
        plt.plot(mean_psth, color=colors[behavior], lw=.5)
        plt.fill_between(np.arange(minute), y1=mean_psth - stderr, y2=mean_psth + stderr, color=colors[behavior],
                         alpha=.3)
        plt.ylabel('Probability', fontsize=10)
        plt.xlabel('Time[s]', fontsize=8)
        plt.xticks([0,  300,  600, 900], [-20, 0,  20,  40], fontsize=8)
        plt.yticks([0, 0.25, .5], [0, 0.25, 0.5], fontsize=8)
        plt.ylim([0, .5])
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
        plt.savefig(output_folder + '/dynamics_'+t+'_' + behaviors[behavior] + '.png', dpi=300)
        plt.savefig(output_folder + '/dynamics_'+t+'_' + behaviors[behavior] + '.pdf', dpi=300)
        plt.close()

#%% Fig 7N
output_folder = root_folder+'/Figures/Optogenetics/A2a_splashTest/Movement_parameters/'
BODY_LCIKING=5
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
m_idx = 0
stim_day=copy.deepcopy(stim_days['splashTest2p2mW'])
first_stim = 3 * minute - RNN_offset
stim_times = np.arange(first_stim, 14 * minute, ISI)
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    laser_times = []
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        laser_times.extend(np.arange(laser_onset,laser_onset+bin_duration,1))
    for behavior in range(num_of_behaviors):
        behavior_indices = np.nonzero(predictions==behavior)[0]
        for idx in behavior_indices:
            if not np.isnan(SSD[idx]):
                if idx in laser_times:
                  laser_on[behavior].append([SSD[idx],velocity[idx]*FPS])
                else:
                    laser_off[behavior].append([SSD[idx],velocity[idx]*FPS])
    m_idx+=1

behavior = BODY_LCIKING
laser_off[behavior] = np.array(laser_off[behavior])
laser_on[behavior] = np.array(laser_on[behavior])
on_percentile = hf.find_Q(laser_on[behavior][:,1],0.75)
on_indices = np.nonzero(laser_on[behavior][:,1]<on_percentile)[0]
off_percentile = hf.find_Q(laser_off[behavior][:, 1], 0.9)
off_indices = np.nonzero(laser_off[behavior][:, 1] < off_percentile)[0]
b_cdf_on = []
b_cdf_off = []
x_axis = np.linspace(0, 0.004,1000)
for ssd in x_axis:
    b_cdf_on.append(np.count_nonzero(laser_on[behavior][:,0] <= ssd) / len(laser_on[behavior]))
    b_cdf_off.append(np.count_nonzero(laser_off[behavior][off_indices,0] <= ssd) / off_indices.size)
plt.figure(figsize=(1.4,1.5), frameon=False)
plt.plot(x_axis,b_cdf_on,c=colors[behavior],label='On',lw=.5)
plt.plot(x_axis,b_cdf_off,c=colors[behavior],ls='--',label='Off',lw=.5)
plt.legend(frameon=False,fontsize=8,title='Laser',loc='upper left',handlelength=1)
plt.xticks([0,0.002,0.004],[0,2,4],fontsize=8)
plt.gca().annotate('$10^{-3}$', xy=(.85, 0.01), xycoords='axes fraction', fontsize=4)
plt.xlabel('SSD(a.u)',fontsize=10)
plt.yticks([0,0.5,1],[0,0.5,1],fontsize=8)
plt.ylim(0,1)
plt.xlim(0,0.004)
plt.ylabel('CDF',fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
vel_on_mean = np.mean(laser_on[behavior][:, 1])
vel_off_mean =  np.mean(laser_off[behavior][off_indices, 1])
vel_on_stderr = np.sqrt(np.var(laser_on[behavior][:, 1])/laser_on[5].shape[0])
vel_off_stderr = np.sqrt(np.var(laser_off[behavior][off_indices, 1])/off_indices.size)
plt.savefig(output_folder + 'SSD_laser_on_off_' + behaviors[behavior]+'_splashTest2p2mW_cdf.pdf', dpi=300)
plt.savefig(output_folder + 'SSD_laser_on_off_' + behaviors[behavior] + '_splashTest2p2mW_cdf.png', dpi=300)
plt.close()