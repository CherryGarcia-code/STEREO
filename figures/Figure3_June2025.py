#%% Imports and constants
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as hf
import bz2
import pickle
import scipy.stats as stats
import scipy
import copy
import sys
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os
import stats_helper_file as shf

mm=1/25.4
FPS=15
sample_rate=15
second = 15
minute = 60*second
#%% Load raw data
root_folder = '.'
folder = root_folder+'Data/'
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
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
trials = ['saline1', 'saline2', 'saline3', 'cocaine1','cocaine2', 'cocaine3','cocaine4', 'cocaine5']
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
surfaceLicking_lut =  np.array([NO_LICKING,UNDEFINED,FLOOR_LICKING,WALL_LICKING,SELF_LICKING,SELF_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
goFrom_behavior = NO_LICKING
goTo_behavior = FLOOR_LICKING
goFrom_label = behaviors_labels[goFrom_behavior]
goTo_label = behaviors_labels[goTo_behavior]
boi_lut = surfaceLicking_lut
#%% Fig 3A
output_folder = 'output/Photometry/forPaper/Full_traces/'
colors_prime = ['#808080','#FFA500','#FF8C00','#FF6347','#E60000','#990000',"#8e63b8"]
sig_type='z'
smooth_window=3
window = np.arange(8*minute,18*minute,1)
# c='controls'
# m = 'c548m10'
c = 'a2a_hm3dq'
m = 'cA242m5'
# output_folder_c =output_folder+'Drd1/'
output_folder_c =output_folder+'A2a/'
fig,ax = plt.subplots(nrows = 8,ncols=1,frameon=False,sharex='all',sharey='all',figsize=(70*mm,70*mm))
colors_gradient =['gray','gray']
colors_gradient.extend(colors_prime)
tr_idx=0
for tr in ['saline1','saline2','saline3','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']:
    # trace = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window) #c58m10
    trace = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)  # cA242m5
    ax[tr_idx].plot(trace[window], label='Left hemisphere',  color=colors_gradient[tr_idx],lw=.5)

    ax[tr_idx].spines['right'].set_visible(False)
    ax[tr_idx].spines['top'].set_visible(False)
    ax[tr_idx].spines['left'].set_visible(False)
    ax[tr_idx].spines['bottom'].set_visible(False)
    if tr_idx<7:
        plt.sca(ax[tr_idx])
        plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        plt.sca(ax[7])
        plt.xticks([3 * minute], [''])
    tr_idx+=1
plt.savefig(output_folder_c+m+'.png',bbox_inches='tight',dpi=300)
plt.savefig(output_folder_c + m+'.pdf',bbox_inches='tight', dpi=300)
plt.close()
#%% Fig 3B,C,E,F
trials_subset = grouped_trials[:-1]
goFrom_dur = 5
smooth_signal = False
smooth_window = 3
output_folder = 'output/Photometry/forPaper/'
colors_prime = ['#808080','#FFA500','#FF8C00','#FF6347','#E60000','#990000']
prom = {'drd1':{},'a2a': {}}
IEI = {'drd1': {},'a2a': {}}
width = {'drd1': {},'a2a': {}}
prom_dist = {'drd1':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]},
             'a2a':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]}}
IEI_dist = {'drd1':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]},
             'a2a':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]}}
prom_mean = {'drd1':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]},
             'a2a':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]}}
IEI_mean = {'drd1':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]},
             'a2a':{'saline':[],'cocaine1':[],'cocaine2':[],'cocaine3':[],'cocaine4':[],'cocaine5':[]}}
trials_subset = ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']
seconds = 15
bout_onset = 5*seconds
sig_type = 'z'
min_prom = 2
min_width = 5
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
        for tr in trials:
            if tr not in CMT[c][m] or 'photom' not in CMT[c][m][tr] : continue
            if 'saline' in tr: tr_label = 'saline'
            else: tr_label = tr
            if tr not in prom[pathway]:
                prom[pathway][tr_label] = []
                IEI[pathway][tr_label] = []
                width[pathway][tr_label] = []
            if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                side = 'both'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
                    right = CMT[c][m][tr]['photom']['right'][sig_type]
                left_peaks,left_peaks_features = scipy.signal.find_peaks(left,prominence = min_prom,width=min_width)
                IEI[pathway][tr_label].extend(list(np.array(np.diff(left_peaks))/seconds))
                prom[pathway][tr_label].extend(left_peaks_features['prominences'])
                width[pathway][tr_label].extend(list(np.array(left_peaks_features['widths'])/seconds))
                right_peaks, right_peaks_features = scipy.signal.find_peaks(right, prominence=min_prom,width=min_width)
                IEI[pathway][tr_label].extend(list(np.array(np.diff(right_peaks))/seconds))
                prom[pathway][tr_label].extend(right_peaks_features['prominences'])
                width[pathway][tr_label].extend(list(np.array(right_peaks_features['widths'])/seconds))
                mouse_proms = list(left_peaks_features['prominences'])
                mouse_proms.extend(right_peaks_features['prominences'])
                mouse_IEI = list(np.array(np.diff(left_peaks))/seconds)
                mouse_IEI.extend(list(np.array(np.diff(right_peaks))/seconds))

            elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                side = 'left'
                if smooth_signal:
                    left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)
                else:
                    left = CMT[c][m][tr]['photom']['left'][sig_type]

                left_peaks,left_peaks_features = scipy.signal.find_peaks(left,prominence = min_prom,width=min_width)
                IEI[pathway][tr_label].extend(list(np.array(np.diff(left_peaks))/seconds))
                prom[pathway][tr_label].extend(left_peaks_features['prominences'])
                width[pathway][tr_label].extend(list(np.array(left_peaks_features['widths'])/seconds))
                mouse_proms = list(left_peaks_features['prominences'])
                mouse_IEI = list(np.array(np.diff(left_peaks))/seconds)

            else:
                side = 'right'
                if smooth_signal:
                    right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)
                else:
                    right = CMT[c][m][tr]['photom']['right'][sig_type]

                right_peaks, right_peaks_features = scipy.signal.find_peaks(right, prominence=min_prom,width=min_width)
                IEI[pathway][tr_label].extend(list(np.array(np.diff(right_peaks))/seconds))
                prom[pathway][tr_label].extend(right_peaks_features['prominences'])
                width[pathway][tr_label].extend(list(np.array(right_peaks_features['widths'])/seconds))
                mouse_proms = list(right_peaks_features['prominences'])
                mouse_IEI = list(np.array(np.diff(right_peaks))/seconds)
            prom_dist[pathway][tr_label].append(mouse_proms)
            IEI_dist[pathway][tr_label].append(mouse_IEI)
            prom_mean[pathway][tr_label].append(np.mean(mouse_proms))
            IEI_mean[pathway][tr_label].append(np.mean(mouse_IEI))
titles = ['Inter Event Interval','Event Prominences','Event Width']
labels = ['IEI(s)','Prominences (Z)','Widths (s)']
lims = [40,12,5]
xticks = [[0,20,40],[0,6,12],[0,2.5,5]]
f_idx=0
for feature in [IEI,prom,width]:
    for pathway in ['drd1','a2a']:
        plt.figure(figsize=(30*mm,23*mm),frameon=False)
        tr_idx=0
        for tr in trials_subset:
            sns.ecdfplot(feature[pathway][tr],color = colors_prime[tr_idx],label = trials_subset[tr_idx],lw=.5)
            plt.xlim(0,lims[f_idx])
            # plt.ylim(0.5, 1)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.ylabel('')
            plt.yticks([0,.5, 1], [0,.5, 1], fontsize=8)
            plt.xticks(xticks[f_idx], xticks[f_idx], fontsize=8)
            tr_idx += 1

        plt.savefig(output_folder + titles[f_idx].replace(' ', '_') + pathway+'.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_folder + titles[f_idx].replace(' ', '_') + pathway+'.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    f_idx+=1

"""Statistics - GLMM ;  Event-level model"""

rows = []
pathway='drd1'
for day in ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']:
    for mouse_idx, mouse_proms in enumerate(prom_dist[pathway][day]):
        for prom_val in mouse_proms:
            rows.append({
                'mouse'   : mouse_idx,  # any unique ID
                'day'     : day,
                'pathway' : pathway,
                'prom'    : prom_val
            })
df = pd.DataFrame(rows)
# Reference category = saline
df['day'] = pd.Categorical(
    df['day'],
    categories=['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5'],
    ordered=False        # keep as nominal; set True if you need < ordering
)

model = smf.mixedlm("prom ~ C(day)",  # fixed effect: categorical day
                    df,
                    groups="mouse")    # random intercept per mouse
result = model.fit(reml=False)
print('(3B) dSPN - Prominence')
print(result.summary())


rows = []
for day in ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']:
    for mouse_idx, mouse_IEI in enumerate(IEI_dist[pathway][day]):
        for IEI_val in mouse_IEI:
            rows.append({
                'mouse'   : mouse_idx,  # any unique ID
                'day'     : day,
                'pathway' : pathway,
                'IEI'    : IEI_val
            })
df = pd.DataFrame(rows)
# Reference category = saline
df['log_IEI'] = np.log(df['IEI'])
df['day'] = pd.Categorical(df['day'],categories=['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5'],ordered=False)

model = smf.mixedlm("log_IEI ~ C(day)",df,groups="mouse")
result = model.fit(reml=False)

print('(3C) dSPN - IEI')
print(result.summary())
shf.validate_lmm(result) # as IEI has an exponential distribution and not normal

#%%





rows = []
pathway='a2a'
for day in ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']:
    for mouse_idx, mouse_proms in enumerate(prom_dist[pathway][day]):
        for prom_val in mouse_proms:
            rows.append({
                'mouse'   : mouse_idx,  # any unique ID
                'day'     : day,
                'pathway' : pathway,
                'prom'    : prom_val
            })
df = pd.DataFrame(rows)
# Reference category = saline
df['day'] = pd.Categorical(
    df['day'],
    categories=['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5'],
    ordered=False        # keep as nominal; set True if you need < ordering
)
# df['log_prom'] = np.log(df['prom'])
model = smf.mixedlm("prom ~ C(day)",  # fixed effect: categorical day
                    df,
                    groups="mouse")    # random intercept per mouse
result = model.fit(reml=False)
print('(3E) iSPN - Prominence')
print(result.summary())


rows = []
for day in ['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']:
    for mouse_idx, mouse_IEI in enumerate(IEI_dist[pathway][day]):
        for IEI_val in mouse_IEI:
            rows.append({
                'mouse'   : mouse_idx,  # any unique ID
                'day'     : day,
                'pathway' : pathway,
                'IEI'    : IEI_val
            })
df = pd.DataFrame(rows)
# Reference category = saline
df['day'] = pd.Categorical(
    df['day'],
    categories=['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5'],
    ordered=False
)

model = sm.MixedLM.from_formula("IEI ~ C(day)",
                    df,
                    groups="mouse",
                    family=sm.families.Gamma(sm.families.links.log())) # IEI is positive & skewed--> Gamma with log link
result = model.fit(method='lbfgs')
print('(3F) iSPN - IEI')
print(result.summary())

#%% Fig 3G
output_folder = 'output/Photometry/forPaper/behavior_overlaid_on_signal/'
sig_type='z'
smooth_window=3
tr = ['splashTest']
c = 'a2a_hm3dq'
# c = 'controls'
m = 'cA184m7'
# m= 'c548m10'
tr = 'cocaine5'
# window = np.arange(5*minute,9*minute,1)
window = np.arange(13*minute,17*minute,1)
photom_offset = CMT[c][m][tr]['photom']['offset']
predictions = np.copy(CMT[c][m][tr]['merged']['predictions']['smartMerge'])[window]
plt.figure(figsize=(75 * mm, 30 * mm))
# left = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['left'][sig_type]), smooth_window)[window]
right = hf.smoothing(np.copy(CMT[c][m][tr]['photom']['right'][sig_type]), smooth_window)[window]
# plt.plot(left, label='Left hemisphere', alpha=.5, c='green',lw=.5)
plt.vlines(x=np.argwhere(predictions==2)[:,0],ymin=-2,ymax=6,linewidth=1.0/15.0,colors=colors[2])
plt.plot(right, label='Right hemisphere', c='#00FFFF',lw=.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xticks([1*minute],[''])
plt.savefig(output_folder+c+'_'+m+'_'+tr+'.png',bbox_inches='tight',dpi=300)
plt.savefig(output_folder+c+'_'+m+'_'+tr+'.pdf',bbox_inches='tight', dpi=300)
plt.close()
#%% Load Ca event-associated bouts
sys.modules[__name__].__dict__.clear()
ifile = bz2.BZ2File('output/Photometry/forPaper/finalManuscript_V1_Apr25/NoLicking2FloorLicking/master_table.pkl', 'rb')
master_table = pickle.load(ifile)
ifile.close()
c_dspns = '#d73027'
c_ispns = '#721515'
seconds=15
pre_switch_window=3*seconds
post_switch_window= 5*seconds
window_length = pre_switch_window+post_switch_window
#%% Fig 3H
SRTs = {'dSPN':{'saline':{'traces':[],'bout_durations':[]},'cocaine':{'traces':[],'bout_durations':[]}},
        'iSPN':{'saline':{'traces':[],'bout_durations':[]},'cocaine':{'traces':[],'bout_durations':[]}}}
SRTs['dSPN']['saline']['traces'] = copy.deepcopy(master_table['saline']['dSPN']['up_regulated']['signal'])
SRTs['dSPN']['saline']['bout_durations'] = copy.deepcopy(master_table['saline']['dSPN']['up_regulated']['duration'])
SRTs['iSPN']['saline']['traces'] = copy.deepcopy(master_table['saline']['iSPN']['up_regulated']['signal'])
SRTs['iSPN']['saline']['bout_durations'] = copy.deepcopy(master_table['saline']['iSPN']['up_regulated']['duration'])

for tr in ['cocaine3','cocaine4','cocaine5']:
    SRTs['dSPN']['cocaine']['traces'].extend(master_table[tr]['dSPN']['up_regulated']['signal'])
    SRTs['dSPN']['cocaine']['bout_durations'].extend(master_table[tr]['dSPN']['up_regulated']['duration'])

    SRTs['iSPN']['cocaine']['traces'].extend(master_table[tr]['iSPN']['up_regulated']['signal'])
    SRTs['iSPN']['cocaine']['bout_durations'].extend(master_table[tr]['iSPN']['up_regulated']['duration'])
fig,ax = plt.subplots(ncols=2,nrows=3,figsize=(65*mm,55*mm),gridspec_kw={'height_ratios':[3,1,1]})
tr_idx=0
for tr in ['saline','cocaine']:
    dSPN = np.array(SRTs['dSPN'][tr]['traces']['up_regulated']) # If this row causes trouble- the signals are not in the same length
    dSPN_dur = np.array(SRTs['dSPN'][tr]['bout_durations']['up_regulated'])
    sorted_indices = np.argsort(dSPN_dur)
    dSPN = dSPN[sorted_indices,:]
    mean = np.mean(dSPN,axis=0)
    err = np.sqrt(np.var(dSPN,axis=0)/dSPN.shape[0])
    ax[0,tr_idx].plot(mean,c=c_dspns,label='dSPN',lw=.7)
    ax[0,tr_idx].fill_between(x=np.arange(mean.size),y1= mean-err,y2=mean+err,color=c_dspns,alpha=.3)
    ax[0,tr_idx].tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.sca(ax[1, tr_idx])
    sns.heatmap(dSPN, ax=ax[1, tr_idx], cbar=False, cmap='magma', vmin=-1,vmax=2.5)
    plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.yticks([dSPN.shape[0]], [dSPN.shape[0]],va='bottom',fontsize=8)
    iSPN = np.array(SRTs['iSPN'][tr]['traces']['up_regulated'])# If this row causes trouble- the signals are not in the same length
    iSPN_dur = np.array(SRTs['iSPN'][tr]['bout_durations']['up_regulated'])
    sorted_indices = np.argsort(iSPN_dur)
    iSPN = iSPN[sorted_indices, :]
    mean = np.mean(iSPN, axis=0)
    err = np.sqrt(np.var(iSPN, axis=0) / iSPN.shape[0])
    ax[0,tr_idx].plot(mean, c=c_ispns,label='iSPN',lw=.7)
    ax[0,tr_idx].fill_between(x=np.arange(mean.size),y1=mean - err, y2=mean + err, color=c_ispns, alpha=.3)
    plt.sca(ax[2, tr_idx])
    sns.heatmap(iSPN, ax=ax[2, tr_idx], vmin=-1,vmax=2.5,cbar=False, cmap='magma')
    plt.xticks(np.arange(0,window_length,FPS),((np.arange(0,window_length,FPS)-pre_switch_window)/FPS).astype(int),rotation=0,fontsize=8)
    plt.yticks([iSPN.shape[0]],[iSPN.shape[0]],va='bottom',fontsize=8)
    ax[0,tr_idx].vlines(x=45,ymin=-1,ymax=2.5 , ls='--',colors='k',lw=1)
    ax[0,tr_idx].set_ylim([-1,3])
    if tr_idx==0:
        ax[0,tr_idx].spines['right'].set_visible(False)
        ax[0,tr_idx].spines['top'].set_visible(False)
    else:
        ax[0, tr_idx].spines[:].set_visible(False)
    if tr_idx>0: ax[0,tr_idx].tick_params(axis='y', which='both', left=False, labelleft=False)
    tr_idx+=1

ax[0,0].legend(frameon=False,fontsize=8)
ax[0,0].set_ylabel('Mean $\Delta$F/F(Z)',fontsize=10)
ax[0,0].set_yticks([0,1.5,3],[0,1.5,3], fontsize=8)
ax[1,0].set_ylabel('dSPN',fontsize=10)
ax[2,0].set_ylabel('iSPN',fontsize=10)
plt.tight_layout(rect=[0.02, 0.02, .99, 0.99], pad=0.1)
plt.savefig('output/Photometry/forPaper/Summary/No_licking2Floor_licking/cocaine_grouped.png',dpi=300)
plt.savefig('output/Photometry/forPaper/Summary/No_licking2Floor_licking/cocaine_grouped.pdf',dpi=300)
plt.close()
#%% Fig 3I
SRTs = {'dSPN':{'saline':{'traces':[],'bout_durations':[]},'cocaine':{'traces':[],'bout_durations':[]}},
        'iSPN':{'saline':{'traces':[],'bout_durations':[]},'cocaine':{'traces':[],'bout_durations':[]}}}
SRTs['dSPN']['saline']['traces'] = copy.deepcopy(master_table['saline']['dSPN']['up_regulated']['signal'])
SRTs['dSPN']['saline']['bout_durations'] = copy.deepcopy(master_table['saline']['dSPN']['up_regulated']['duration'])
SRTs['iSPN']['saline']['traces'] = copy.deepcopy(master_table['saline']['iSPN']['up_regulated']['signal'])
SRTs['iSPN']['saline']['bout_durations'] = copy.deepcopy(master_table['saline']['iSPN']['up_regulated']['duration'])

for tr in ['cocaine3','cocaine4','cocaine5']:
    SRTs['dSPN']['cocaine']['traces'].extend(master_table[tr]['dSPN']['up_regulated']['signal'])
    SRTs['dSPN']['cocaine']['bout_durations'].extend(master_table[tr]['dSPN']['up_regulated']['duration'])

    SRTs['iSPN']['cocaine']['traces'].extend(master_table[tr]['iSPN']['up_regulated']['signal'])
    SRTs['iSPN']['cocaine']['bout_durations'].extend(master_table[tr]['iSPN']['up_regulated']['duration'])


dSPN_traces = copy.deepcopy(SRTs['dSPN']['cocaine']['traces'])
dSPN_durations = np.array(SRTs['dSPN']['cocaine']['bout_durations'])
iSPN_traces = copy.deepcopy(SRTs['iSPN']['cocaine']['traces'])
iSPN_durations = np.array(SRTs['iSPN']['cocaine']['bout_durations'])
colors_edge=[c_dspns,c_ispns]
plt.figure(figsize=(60*mm,60*mm),frameon=False)
ax = sns.violinplot(data=[dSPN_durations,iSPN_durations],palette=['white','white'],inner=None,linewidth=.8)
for i, violin in enumerate(ax.collections):  # ::2 skips the inner fill poly collections
    violin.set_edgecolor(colors_edge[i])
plt.scatter(np.zeros_like(dSPN_durations),dSPN_durations,marker='o',s=2,color='white',linewidths=.3,edgecolor=c_dspns)
plt.scatter(np.ones_like(iSPN_durations),iSPN_durations,marker='o',s=2,color='white',linewidths=.3,edgecolor=c_ispns)
plt.xticks([0,1],['dSPN','iSPN'],fontsize=8)
plt.ylabel('Bout duration(s)',fontsize=10)
plt.yticks(np.arange(0,751,150),np.arange(0,51,10),fontsize=8)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('output/Photometry/forPaper/finalManuscript_V1_Apr25/NoLicking2FloorLicking/timeWarped/Bout_duration.png',dpi=300)
plt.savefig('output/Photometry/forPaper/finalManuscript_V1_Apr25/NoLicking2FloorLicking/timeWarped/Bout_duration.pdf',dpi=300)
plt.close()

target_bout_length = 2*seconds
tail_length = 2*seconds
dSPN_timeWarped = np.zeros((len(dSPN_traces),pre_switch_window+target_bout_length+tail_length))
iSPN_timeWarped = np.zeros((len(iSPN_traces),pre_switch_window+target_bout_length+tail_length))
plt.figure(figsize=(65*mm,60*mm))
for row in range(len(dSPN_traces)):
    baseline_seg = dSPN_traces[row][:pre_switch_window]
    bout_seg = dSPN_traces[row][pre_switch_window:pre_switch_window+dSPN_durations[row]]
    post_bout_seg = dSPN_traces[row][pre_switch_window + dSPN_durations[row]:]
    timeWarped_bout_seg = np.interp(x=np.linspace(0, len(bout_seg) - 1, int(target_bout_length)),xp=np.arange(len(bout_seg)),fp=bout_seg)
    timeWarped_full_segment = np.hstack([baseline_seg, timeWarped_bout_seg,post_bout_seg])
    dSPN_timeWarped[row] = np.copy(timeWarped_full_segment)

mean = np.nanmean(dSPN_timeWarped,axis=0)
err = np.sqrt(np.nanvar(dSPN_timeWarped, axis=0)/dSPN_timeWarped.shape[0])
plt.plot(mean,color=c_dspns,lw=.7,label='dSPN')
plt.fill_between(x= np.arange(mean.size),y1=mean-err,y2=mean+err,alpha=.3,color=c_dspns)

for row in range(len(iSPN_traces)):
    baseline_seg = iSPN_traces[row][:pre_switch_window]
    bout_seg = iSPN_traces[row][pre_switch_window:pre_switch_window + iSPN_durations[row]]
    post_bout_seg = iSPN_traces[row][pre_switch_window + iSPN_durations[row]:]
    timeWarped_bout_seg = np.interp(x=np.linspace(0, len(bout_seg) - 1, int(target_bout_length)),xp=np.arange(len(bout_seg)),fp=bout_seg)
    timeWarped_full_segment = np.hstack([baseline_seg, timeWarped_bout_seg,post_bout_seg])
    iSPN_timeWarped[row] = np.copy(timeWarped_full_segment)
mean = np.nanmean(iSPN_timeWarped,axis=0)
err = np.sqrt(np.nanvar(iSPN_timeWarped, axis=0)/iSPN_timeWarped.shape[0])
plt.plot(mean,color=c_ispns,lw=.7,label='iSPN')
plt.fill_between(x= np.arange(mean.size),y1=mean-err,y2=mean+err,alpha=.3,color=c_ispns)
plt.hlines(xmin=pre_switch_window,xmax=pre_switch_window+target_bout_length,y=1.3,lw=2,colors="#d73027")
plt.legend(fontsize=8,frameon=False)
tot_window = pre_switch_window+target_bout_length+tail_length
plt.xticks(np.arange(0,tot_window+1,15),['S-3','S-2','S-1','S','','E','E+1','E+2'],fontsize=8)
plt.ylim([-.5,1.5])
plt.yticks([0,.5,1,1.5],[0,.5,1,1.5],fontsize=8)
plt.ylabel('Mean $\Delta$F/F$_0$(Z)')
plt.tight_layout()
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig('output/Photometry/forPaper/finalManuscript_V1_Apr25/NoLicking2FloorLicking/timeWarped/timeWarped.png',dpi=300)
plt.savefig('output/Photometry/forPaper/finalManuscript_V1_Apr25/NoLicking2FloorLicking/timeWarped/timeWarped.pdf',dpi=300)
plt.close()
