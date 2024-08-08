import numpy as np
import bz2
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
from cycler import cycler
import helper_functions as hf
import scipy.signal as signal
import pickle
from matplotlib.collections import LineCollection
mpl.rcParams["mathtext.fontset"] = 'cm'
import scipy.stats as stats
root_folder = 'May24/'
folder = root_folder+'Data/'
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163','#969595', '#b4a7d6']
SSD_color = '#CC6677'
velocity_color = '#44AA99'
FPS=15
sample_rate=15
ifile = bz2.BZ2File(folder + 'CTM_May24.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT_May24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
trials  = ['saline1','saline2','saline3','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5','splashTest']
num_of_trials = len(trials)
num_of_behaviors = len(behaviors)
del CMT['a2a_hm3dq']['cA184m7']['cocaine4']['photom']['right']
del CMT['a2a_hm3dq']['cA242m6']['saline1']['photom']['right']
del CMT['controls']['cA242m9']['saline2']['photom']['right']
mice_list = ['c548m1','c548m10','c548m11','c548m8','cA242m4','cA242m9','cA184m4','cA184m7','cA242m5','cA242m6','cA242m8']
sig_type = 'z' # or z
#%% Heatmaps and means of behavior related transients (BRTs)
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Summary'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={'long':[],'short':[]}
            bout_length[pathway][b][tr]={'long':[],'short':[]}
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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
                    right = CMT[c][m][tr]['photom']['right'][sig_type]
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left'][sig_type]
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right'][sig_type]

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                    for length in ['short', 'long']:
                        for bout_idx in range(len(np_bouts[length])):
                            # print(len(np_bouts[length]) , bout_idx)
                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':
                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_left)

                            elif side == 'right':
                                if np_bouts[length][bout_idx][0] >= 5*seconds and right.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_right = right[np_bouts[length][bout_idx][0] - 5 * seconds:np_bouts[length][bout_idx][0] + 10 * seconds] - np.mean(
                                        right[np_bouts[length][bout_idx][0] - 5 * seconds:np_bouts[length][bout_idx][0] - 2 * seconds])
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_right)
                            else:
                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])


                                    sig_right = right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds] - np.mean(
                                        right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])

                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(sig_left)
                                    photometry[pathway][b][tr][length].append(sig_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6,7]
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, sharex=True, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            long = np.array(photometry[c][b][tr]['long'])
            short = np.array(photometry[c][b][tr]['short'])
            if long.shape[0] ==0 and short.shape[0]==0 :
                tr_idx+=1
                continue
            n_long = long.shape[0]
            if long.shape[0]>0:
                mean = np.nanmean(long,axis=0)
                stderr = np.sqrt(np.nanvar(long, axis=0)/long.shape[0])
                ax[0,tr_idx].plot(mean, label='Long (n='+str(n_long)+')',c=c1, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c1, alpha=low_alpha)

            n_short = short.shape[0]
            if short.shape[0]>0:
                mean = np.nanmean(short,axis=0)
                stderr = np.sqrt(np.nanvar(short, axis=0)/short.shape[0])
                ax[0, tr_idx].plot(mean, label='Short (n='+str(n_short)+')', c=c2, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c2, alpha=high_alpha)
            n_str = 'N$_{short}$='+str(n_short)+', N$_{long}$='+str(n_long)
            print(long.shape, short.shape)
            if short.shape[0]>0 and long.shape[0] == 0:
                all_events = np.copy(short)
            elif short.shape[0] == 0 and long.shape[0]>0:
                all_events = np.copy(long)
            else:
                all_events = np.vstack((short,long))
            if tr_idx==0:
                ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z-score)',fontsize=12)
            ax[0,tr_idx].set_ylim(lims[sig_type][c])
            ax[0,tr_idx].vlines(x=bout_onset, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            ax[0,tr_idx].legend()
            plt.sca(ax[0,tr_idx])
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')
            bout_length[c][b][tr]['short'].extend(bout_length[c][b][tr]['long'])
            events_length = np.array(bout_length[c][b][tr]['short'])
            indices = np.argsort(events_length)
            # print(long.shape , short.shape , all_events.shape , indices.size , events_length.shape)
            ax[1, tr_idx] = sns.heatmap(all_events[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds),
                       ((np.arange(0, window_length + 1, 2 * seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1
        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.suptitle(c+';  $\\theta$$_{short}^{long}$ = 2s ; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_'+sig_type+'.png',dpi=300)
        plt.close()



#%% Per mouse transients
cohort = 'drd1'
if cohort == 'drd1':
    cohorts = ['drd1_hm4di','drd1_hm3dq','controls']
else:
    cohorts = ['a2a_hm4di','a2a_hm3dq','controls']
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/perMouse/'+cohort+'/'
all_mice_output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Allmice/'+cohort+'/'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 1
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 10*seconds
window_right_wing = 5*seconds # relative to bout onset
window_left_wing = 5*seconds # relative to bout onset
bout_onset = 5*seconds
behaviors_prime = [0,1,2,3,4,6,7]
length = 'long'
for b in behaviors_prime:
    photometry = {}
    bout_length = {}
    for tr in ['saline1', 'saline2', 'saline3']:
        print(tr)
        for c in cohorts:
            if tr not in CTM[c]: continue
            for m in CTM[c][tr]:
                if cohort=='drd1':
                    if  c =='controls' and m.startswith('cA'):continue # for drd1
                else:
                    if c == 'controls' and not m.startswith('cA'): continue  # for a2a
                if not 'photom' in CMT[c][m][tr].keys():continue
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom']:
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']
                predictions = CMT[c][m][tr]['merged']
                if m not in photometry :
                    photometry[m] = []
                    bout_length[m] = []
                np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                for bout_idx in range(len(np_bouts[length])):
                    if side == 'left':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)

                    elif side == 'right':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and right.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_right = right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_right)
                    else:
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            sig_right = right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)
                            photometry[m].append(sig_right)
    num_of_mice = len(photometry.keys())

    for m in photometry:
        if len(photometry[m]) == 0:
            num_of_mice -= 1
        else:
            transients = np.copy(np.array(photometry[m]))
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0:  # in  case all rows were nan moues should be omitted
                num_of_mice -= 1
    if num_of_mice > 0:
        hr = [3]
        for m_idx in range(num_of_mice):
            hr.append(1)
        fig , ax = plt.subplots(nrows = num_of_mice+1 , sharex = 'all' ,gridspec_kw={'height_ratios': hr}, figsize=(8,12))
        row_idx=1
        all_mice_transients = np.zeros(window_length)
        all_mice_lengths = []
        for m in photometry:
            events_length = np.array(bout_length[m])
            indices = np.argsort(events_length)
            transients = np.array(photometry[m])
            if transients.shape[0] == 0: continue
            transients = transients[indices,]
            all_mice_transients = np.vstack([all_mice_transients, transients])
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0: # in  case all rows were nan moues should be ommited
                continue
            all_mice_lengths.extend(bout_length[m])

            plt.sca(ax[row_idx])
            ax[row_idx] = sns.heatmap(transients, vmin=lims[sig_type][cohort][0], vmax=lims[sig_type][cohort][1], ax=ax[row_idx], cbar=False)
            plt.yticks(np.linspace(0,transients.shape[0],3,dtype=int),np.linspace(0,transients.shape[0],3,dtype=int))
            plt.ylabel(m)
            plt.sca(ax[0])
            mean = np.nanmean(transients,axis=0)
            stderr = np.sqrt(np.nanvar(transients,axis=0)/transients.shape[0])
            plt.plot(mean,label = m,alpha=high_alpha)
            plt.fill_between(x = np.arange(window_length),y1 = mean-stderr,y2=mean+stderr,alpha=low_alpha)
            plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
            plt.ylabel('$\Delta{F}/{F}(Z)$')
            row_idx+=1
        plt.sca(ax[0])

        plt.legend(loc = 'upper left' , ncol = int(np.ceil(num_of_mice/2)))
        plt.sca(ax[num_of_mice])
        plt.xticks(np.arange(0, window_length + 1,  seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.xlabel('Time relative to bout onset')
        plt.suptitle('Saline - ' + behaviors[b])
        plt.savefig(output_folder+'saline/'+behaviors[b]+'.png')
        plt.close()

    all_mice_events_length = np.array(all_mice_lengths)
    sorted_indices = np.argsort(all_mice_events_length)
    all_mice_transients = all_mice_transients[1:,:]
    all_mice_transients = all_mice_transients[sorted_indices,]
    all_mice_transients =  all_mice_transients[~np.isnan( all_mice_transients).all(axis=1)]
    mean = np.nanmean(all_mice_transients,axis=0)
    stderr = np.sqrt(np.nanvar(all_mice_transients,axis=0)/all_mice_transients.shape[0])
    fig,ax = plt.subplots(nrows=2,sharex='all')
    plt.sca(ax[1])
    ax[1] = sns.heatmap(all_mice_transients, vmin=lims[sig_type][cohort][0], vmax=lims[sig_type][cohort][1], ax=ax[1], cbar=False)
    plt.yticks(np.linspace(0, all_mice_transients.shape[0], 3, dtype=int), np.linspace(0, all_mice_transients.shape[0], 3, dtype=int))
    plt.xticks(np.arange(0, window_length + 1, seconds),
               ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
    plt.xlabel('Time relative to bout onset')
    plt.ylabel('# of events')
    plt.sca(ax[0])
    plt.plot(mean, alpha=high_alpha)
    plt.fill_between(x=np.arange(window_length), y1=mean - stderr, y2=mean + stderr, alpha=low_alpha)
    plt.ylim(lims[sig_type][cohort])
    plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
    plt.ylabel('$\Delta{F}/{F}(Z)$')
    plt.suptitle(cohort + ' ; Saline Days ; '+behaviors[b] )
    plt.savefig(all_mice_output_folder + 'saline/' + behaviors[b] + '.png')
    plt.close()


for tr in ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5', 'splashTest']:
    print(tr)
    for b in behaviors_prime:
        photometry = {}
        bout_length = {}
        for c in cohorts:
            if tr not in CTM[c]: continue
            for m in CTM[c][tr]:
                if cohort=='drd1':
                    if c =='controls' and m.startswith('cA') : continue # for drd1
                else:
                    if c == 'controls' and not m.startswith('cA'): continue  # for a2a
                if not 'photom' in CMT[c][m][tr].keys():continue
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom']:
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom']:
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']
                predictions = CMT[c][m][tr]['merged']
                photometry[m] = []
                bout_length[m] = []
                np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                for bout_idx in range(len(np_bouts[length])):
                    if side == 'left':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[
                                       np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)

                    elif side == 'right':
                        if np_bouts[length][bout_idx][0] >= window_left_wing and right.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_right = right[
                                        np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_right)
                    else:
                        if np_bouts[length][bout_idx][0] >= window_left_wing and left.shape[0] >= np_bouts[length][bout_idx][0] + window_right_wing:
                            sig_left = left[
                                       np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                left[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            sig_right = right[
                                        np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] + window_right_wing] - np.mean(
                                right[np_bouts[length][bout_idx][0] - window_left_wing:np_bouts[length][bout_idx][0] - 2 * seconds])

                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            bout_length[m].append(len(np_bouts[length][bout_idx]))
                            photometry[m].append(sig_left)
                            photometry[m].append(sig_right)

        num_of_mice = len(photometry.keys())
        num_of_events = 0
        for m in photometry:
            num_of_events+=len(photometry[m])
            if len(photometry[m]) == 0:
                num_of_mice -= 1
            else:
                transients = np.copy(np.array(photometry[m]))
                transients = transients[~np.isnan(transients).all(axis=1)]
                if transients.shape[0] == 0:  # in  case all rows were nan moues should be omitted
                    num_of_mice -= 1
        print(tr,behaviors[b],num_of_events)
        if num_of_mice==0: continue
        hr = [3]
        for m_idx in range(num_of_mice):
            hr.append(1)
        all_mice_transients = np.zeros(window_length)
        all_mice_lengths = []
        fig , ax = plt.subplots(nrows = num_of_mice+1,sharex = 'all',gridspec_kw={'height_ratios': hr}, figsize=(8,12))
        row_idx=1
        for m in photometry:
            events_length = np.array(bout_length[m])
            indices = np.argsort(events_length)
            transients = np.array(photometry[m])
            if transients.shape[0] == 0: continue
            transients = transients[indices,]
            all_mice_transients = np.vstack([all_mice_transients, transients])
            transients = transients[~np.isnan(transients).all(axis=1)]
            if transients.shape[0] == 0:# in  case all rows were nan moues should be ommited
                continue
            all_mice_lengths.extend(bout_length[m])
            plt.sca(ax[row_idx])
            ax[row_idx] = sns.heatmap(transients, vmin=lims[sig_type][cohort][0], vmax=lims[sig_type][cohort][1], ax=ax[row_idx], cbar=False)
            plt.yticks(np.linspace(0,transients.shape[0],3,dtype=int),np.linspace(0,transients.shape[0],3,dtype=int),rotation='horizontal')
            plt.ylabel(m)
            plt.sca(ax[0])
            mean = np.nanmean(transients,axis=0)
            stderr = np.sqrt(np.nanvar(transients,axis=0)/transients.shape[0])
            plt.plot(mean,label = m,alpha=high_alpha)
            plt.fill_between(x = np.arange(window_length),y1 = mean-stderr,y2=mean+stderr,alpha=low_alpha)
            plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
            plt.ylabel('$\Delta{F}/{F}(Z)$')
            row_idx+=1
        plt.sca(ax[0])
        plt.legend(loc = 'upper left' , ncol = int(np.ceil(num_of_mice/2)))
        plt.sca(ax[num_of_mice])
        plt.xlabel('Time relative to bout onset')
        plt.xticks(np.arange(0, window_length + 1,  seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.suptitle(tr + '-' + behaviors[b])
        plt.savefig(output_folder+tr+'/'+behaviors[b]+'.png')
        plt.close()
        all_mice_events_length = np.array(all_mice_lengths)
        sorted_indices = np.argsort(all_mice_events_length)
        all_mice_transients = all_mice_transients[1:, :]
        all_mice_transients = all_mice_transients[sorted_indices,]
        all_mice_transients = all_mice_transients[~np.isnan(all_mice_transients).all(axis=1)]
        mean = np.nanmean(all_mice_transients, axis=0)
        stderr = np.sqrt(np.nanvar(all_mice_transients, axis=0) / all_mice_transients.shape[0])
        fig, ax = plt.subplots(nrows=2, sharex='all')
        plt.sca(ax[1])
        ax[1] = sns.heatmap(all_mice_transients, vmin=lims[sig_type][cohort][0], vmax=lims[sig_type][cohort][1], ax=ax[1], cbar=False)
        plt.yticks(np.linspace(0, all_mice_transients.shape[0], 3, dtype=int), np.linspace(0, all_mice_transients.shape[0], 3, dtype=int))
        plt.xticks(np.arange(0, window_length + 1, seconds),
                   ((np.arange(0, window_length + 1, seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.xlabel('Time relative to bout onset')
        plt.ylabel('# of events')
        plt.sca(ax[0])
        plt.plot(mean, alpha=high_alpha)
        plt.fill_between(x=np.arange(window_length), y1=mean - stderr, y2=mean + stderr, alpha=low_alpha)
        plt.ylim(lims[sig_type][cohort])
        plt.vlines(x=bout_onset, ymin=plt.ylim()[0], ymax=plt.ylim()[1], ls='dashed', colors='k')
        plt.ylabel('$\Delta{F}/{F}(Z)$')
        plt.suptitle(cohort + ' ; '+ tr+' ; ' + behaviors[b])
        plt.savefig(all_mice_output_folder + tr+'/' + behaviors[b] + '.png')
        plt.close()





#%% Heatmaps and means of behavior related transients (BRTs) from bout onset - z scored over bout
sig_type = 'z_over_bout'
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Summary'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]},'z_over_bout':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={'long':[],'short':[]}
            bout_length[pathway][b][tr]={'long':[],'short':[]}
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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                    for length in ['short', 'long']:
                        for bout_idx in range(len(np_bouts[length])):

                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':

                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left_tot = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds]
                                    baseline_left = np.mean(left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot-baseline_left,nan_policy='omit'))
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_left)

                            elif side == 'right':
                                if np_bouts[length][bout_idx][0] >= 5*seconds and right.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_right_tot = right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds]
                                    baseline_right = np.mean(right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot-baseline_right,nan_policy='omit'))
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_right)
                            else:
                                if np_bouts[length][bout_idx][0] >= 5*seconds and left.shape[0] >= np_bouts[length][bout_idx][0] + 10*seconds:
                                    sig_left_tot = left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds]
                                    baseline_left = np.mean(left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot-baseline_left,nan_policy='omit'))
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))


                                    sig_right_tot = right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] + 10*seconds]
                                    baseline_right = np.mean(right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot-baseline_right,nan_policy='omit'))
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))

                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_left)
                                    photometry[pathway][b][tr][length].append(z_over_bout_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6,7]
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, sharex=True, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            long = np.array(photometry[c][b][tr]['long'])
            short = np.array(photometry[c][b][tr]['short'])
            if long.shape[0] ==0 and short.shape[0]==0 :
                tr_idx+=1
                continue
            n_long = long.shape[0]
            if long.shape[0]>0:
                mean = np.nanmean(long,axis=0)
                stderr = np.sqrt(np.nanvar(long, axis=0)/long.shape[0])
                ax[0,tr_idx].plot(mean, label='Long (n='+str(n_long)+')',c=c1, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c1, alpha=low_alpha)

            n_short = short.shape[0]
            if short.shape[0]>0:
                mean = np.nanmean(short,axis=0)
                stderr = np.sqrt(np.nanvar(short, axis=0)/short.shape[0])
                ax[0, tr_idx].plot(mean, label='Short (n='+str(n_short)+')', c=c2, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c2, alpha=high_alpha)
            n_str = 'N$_{short}$='+str(n_short)+', N$_{long}$='+str(n_long)
            print(long.shape, short.shape)
            if short.shape[0]>0 and long.shape[0] == 0:
                all_events = np.copy(short)
            elif short.shape[0] == 0 and long.shape[0]>0:
                all_events = np.copy(long)
            else:
                all_events = np.vstack((short,long))
            if tr_idx==0:
                ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z-score)',fontsize=12)
            ax[0,tr_idx].set_ylim(lims[sig_type][c])
            ax[0,tr_idx].vlines(x=bout_onset, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            ax[0,tr_idx].legend()
            plt.sca(ax[0,tr_idx])
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')
            bout_length[c][b][tr]['short'].extend(bout_length[c][b][tr]['long'])
            events_length = np.array(bout_length[c][b][tr]['short'])
            indices = np.argsort(events_length)
            # print(long.shape , short.shape , all_events.shape , indices.size , events_length.shape)
            ax[1, tr_idx] = sns.heatmap(all_events[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds),
                       ((np.arange(0, window_length + 1, 2 * seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1
        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.suptitle(c+';  $\\theta$$_{short}^{long}$ = 2s ; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_'+sig_type+'.png',dpi=300)
        plt.close()

#%% Heatmaps and means of behavior related transients (BRTs) from bout end - z scored over bout
sig_type = 'z_over_bout'
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Summary/Termination'
np_sig = {'left':{'short':[],'long':[]},'right':{'short':[],'long':[]}}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 7*seconds
bout_onset = 2*seconds
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]},'z_over_bout':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={'long':[],'short':[]}
            bout_length[pathway][b][tr]={'long':[],'short':[]}
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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    np_bouts, wp_bouts = hf.find_bouts_long_short_np_wp(predictions, b, short_long_threshold=theta)
                    for length in ['short', 'long']:
                        for bout_idx in range(len(np_bouts[length])):

                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':
                                if np_bouts[length][bout_idx][-1] >= 2*seconds and left.shape[0] >= np_bouts[length][bout_idx][-1] + 5*seconds:
                                    sig_left_tot = left[np_bouts[length][bout_idx][-1] - 2*seconds:np_bouts[length][bout_idx][-1] + 5*seconds]
                                    # baseline_left = np.mean(left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot-baseline_left,nan_policy='omit'))
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_left)

                            elif side == 'right':
                                if np_bouts[length][bout_idx][-1] >= 2*seconds and right.shape[0] >= np_bouts[length][bout_idx][-1] + 5*seconds:
                                    sig_right_tot = right[np_bouts[length][bout_idx][-1] - 2*seconds:np_bouts[length][bout_idx][-1] + 5*seconds]
                                    # baseline_right = np.mean(right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot-baseline_right,nan_policy='omit'))
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_right)
                            else:
                                if np_bouts[length][bout_idx][-1] >= 2*seconds and left.shape[0] >= np_bouts[length][bout_idx][-1] + 5*seconds:
                                    sig_left_tot = left[np_bouts[length][bout_idx][-1] - 2*seconds:np_bouts[length][bout_idx][-1] + 5*seconds]
                                    # baseline_left = np.mean(left[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot-baseline_left,nan_policy='omit'))
                                    z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))


                                    sig_right_tot = right[np_bouts[length][bout_idx][-1] - 2*seconds:np_bouts[length][bout_idx][-1] + 5*seconds]
                                    # baseline_right = np.mean(right[np_bouts[length][bout_idx][0] - 5*seconds:np_bouts[length][bout_idx][0] - 2*seconds])
                                    # z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot-baseline_right,nan_policy='omit'))
                                    z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))

                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    bout_length[pathway][b][tr][length].append(len(np_bouts[length][bout_idx]))
                                    photometry[pathway][b][tr][length].append(z_over_bout_left)
                                    photometry[pathway][b][tr][length].append(z_over_bout_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6,7]
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, sharex=True, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            long = np.array(photometry[c][b][tr]['long'])
            short = np.array(photometry[c][b][tr]['short'])
            if long.shape[0] ==0 and short.shape[0]==0 :
                tr_idx+=1
                continue
            n_long = long.shape[0]
            print(c , behaviors[b] ,  long.shape)
            if long.shape[0]>0:
                mean = np.nanmean(long,axis=0)
                stderr = np.sqrt(np.nanvar(long, axis=0)/long.shape[0])
                ax[0,tr_idx].plot(mean, label='Long (n='+str(n_long)+')',c=c1, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c1, alpha=low_alpha)

            n_short = short.shape[0]
            if short.shape[0]>0:
                mean = np.nanmean(short,axis=0)
                stderr = np.sqrt(np.nanvar(short, axis=0)/short.shape[0])
                ax[0, tr_idx].plot(mean, label='Short (n='+str(n_short)+')', c=c2, alpha=line_alpha)
                ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c2, alpha=high_alpha)
            n_str = 'N$_{short}$='+str(n_short)+', N$_{long}$='+str(n_long)
            print(long.shape, short.shape)
            if short.shape[0]>0 and long.shape[0] == 0:
                all_events = np.copy(short)
            elif short.shape[0] == 0 and long.shape[0]>0:
                all_events = np.copy(long)
            else:
                all_events = np.vstack((short,long))
            if tr_idx==0:
                ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z-score)',fontsize=12)
            ax[0,tr_idx].set_ylim(lims[sig_type][c])
            ax[0,tr_idx].vlines(x=bout_onset, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            ax[0,tr_idx].legend()
            plt.sca(ax[0,tr_idx])
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')
            bout_length[c][b][tr]['short'].extend(bout_length[c][b][tr]['long'])
            events_length = np.array(bout_length[c][b][tr]['short'])
            indices = np.argsort(events_length)
            # print(long.shape , short.shape , all_events.shape , indices.size , events_length.shape)
            ax[1, tr_idx] = sns.heatmap(all_events[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds),
                       ((np.arange(0, window_length + 1, 2 * seconds) - 2 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1
        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.suptitle(c+'Signal around termination of bout;  $\\theta$$_{short}^{long}$ = 2s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_'+sig_type+'.png',dpi=300)
        plt.close()

#%% Heatmaps and means of behavior related transients (BRTs) separated based on bout length  - 1-2,2-3,3-4,4-5,>5 - z scored over bout
sig_type = 'z_over_bout'
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/binned_durations/Summary'
times = [1,2,3,4,5]
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
all_events = {}
all_events_length = {}
seconds = 15
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]},'z_over_bout':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    all_events[pathway] = {}
    all_events_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        all_events[pathway][b] = {}
        all_events_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]={1:[],2:[],3:[],4:[],5:[]}
            all_events[pathway][b][tr] = []
            all_events_length[pathway][b][tr] = []

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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']

                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    session_bouts = hf.groupedByDuration_bouts(predictions, b ,times)
                    for dur in times:
                        for bout_idx in range(len(session_bouts[dur])):
                            window_start = session_bouts[dur][bout_idx][0]-2 * seconds
                            if dur != times[-1]:
                                window_end = session_bouts[dur][bout_idx][0]+(dur + 1 + 2) * seconds
                            else:
                                window_end = session_bouts[dur][bout_idx][-1]+2* seconds
                            if 'saline' in tr: tr = 'saline'
                            if side== 'left':
                                if session_bouts[dur][bout_idx][0]>=2*seconds and left.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_left_tot = left[window_start:window_end]
                                    if not np.all(np.isnan(sig_left_tot)):
                                        z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                        photometry[pathway][b][tr][dur].append(z_over_bout_left)
                                        all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                        all_events[pathway][b][tr].append(stats.zscore(left[session_bouts[dur][bout_idx][0] - 2*seconds:session_bouts[dur][bout_idx][0] + 7*seconds],nan_policy='omit'))
                            elif side == 'right':
                                if session_bouts[dur][bout_idx][0]>=2*seconds and right.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_right_tot = right[window_start:window_end]
                                    if not np.all(np.isnan(sig_right_tot)):
                                        z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                        bout_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                        photometry[pathway][b][tr][dur].append(z_over_bout_right)

                                        all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                        all_events[pathway][b][tr].append(stats.zscore(right[session_bouts[dur][bout_idx][0] - 2*seconds:session_bouts[dur][bout_idx][0] + 7*seconds],nan_policy='omit'))
                            else:
                                if session_bouts[dur][bout_idx][0]>=2*seconds and left.shape[0] >= session_bouts[dur][bout_idx][-1] + 2*seconds:
                                    sig_left_tot = left[window_start:window_end]
                                    if not np.all(np.isnan(sig_left_tot)):
                                        z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot, nan_policy='omit'))
                                        photometry[pathway][b][tr][dur].append(z_over_bout_left)
                                        all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                        all_events[pathway][b][tr].append(stats.zscore(left[session_bouts[dur][bout_idx][
                                                                                                0] - 2 * seconds:
                                                                                            session_bouts[dur][bout_idx][
                                                                                                0] + 7 * seconds],
                                                                                       nan_policy='omit'))

                                    sig_right_tot = right[window_start:window_end]
                                    if not np.all(np.isnan(sig_right_tot)):
                                        z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                        photometry[pathway][b][tr][dur].append(z_over_bout_right)
                                        all_events_length[pathway][b][tr].append(len(session_bouts[dur][bout_idx]))
                                        all_events[pathway][b][tr].append(stats.zscore(right[session_bouts[dur][bout_idx][0] - 2 * seconds:session_bouts[dur][bout_idx][0] + 7 * seconds],nan_policy='omit'))

cohorts = ['drd1','a2a']
labels = ['${{1s} \leq {|event|<2s}}$','${{2s} \leq {|event|<3s}}$','${{3s} \leq {|event|<4s}}$','${{4s} \leq {|event|<5s}}$','${{5s} \leq {|event|}}$']
dur_colors = ['#264653','#2A9D8F','#E9C46A','#F4A261','#E76F51']
behaviors_prime = [0,1,2,3,4,6,7]
print('start plotting')
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=len(times)+1, ncols=7, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            if len(all_events_length[c][b][tr])==0:
                for dur_idx in range(len(times)+1):
                    plt.sca(ax[dur_idx, tr_idx])
                    plt.tick_params(left=False, right=False, labelleft=False,
                                    labelbottom=False, bottom=False, top=False)
                    ax[dur_idx, tr_idx].spines["bottom"].set_visible(False)
                    ax[dur_idx, tr_idx].spines['right'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['top'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['left'].set_visible(False)
                tr_idx+=1
                continue
            dur_idx=0
            for dur in times[:-1]:
                print(b,c,tr, dur,len(photometry[c][b][tr][dur]))
                if len(photometry[c][b][tr][dur])>0:
                    transients = np.array(photometry[c][b][tr][dur])
                    num_of_transients = transients.shape[0]
                    plt.sca(ax[dur_idx,tr_idx])
                    mean = np.nanmean(transients,axis=0)
                    stderr = np.sqrt(np.nanvar(transients, axis=0)/num_of_transients)
                    plt.plot(np.arange((dur+5)*seconds),mean, label='$(N='+str(num_of_transients)+')$',c=dur_colors[dur_idx], alpha=line_alpha)
                    plt.fill_between(np.arange((dur+5)*seconds), y1=mean - stderr,y2=mean + stderr, color=dur_colors[dur_idx], alpha=low_alpha)
                    plt.xlim([0,15*seconds])
                    plt.legend()
                    if dur_idx == 0:
                        plt.title(tr + '\n' + labels[dur_idx])
                    else:
                        plt.title(labels[dur_idx])
                    if tr_idx == 0: plt.ylabel('${\Delta F/F (Z)}$')
                    plt.vlines(x=2 * seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                    plt.vlines(x=(dur+1+2)*seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                    colors = ["k", "white"]
                    x = [0, (dur + 5) * seconds]
                    y = [0, 0]
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, colors=colors, linewidth=1, transform=ax[dur_idx, tr_idx].get_xaxis_transform(), clip_on=False)
                    ax[dur_idx, tr_idx].add_collection(lc)
                    ax[dur_idx, tr_idx].spines["bottom"].set_visible(False)
                    ax[dur_idx, tr_idx].spines['right'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['top'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['right'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['top'].set_visible(False)
                    plt.xticks(np.arange(0,(dur+5)*seconds,1*seconds),((np.arange(0,(dur+5)*seconds,1*seconds)-2*seconds)/(1*seconds)).astype(int))
                else:
                    plt.sca(ax[dur_idx, tr_idx])
                    plt.tick_params(left=False, right=False, labelleft=False,
                                    labelbottom=False, bottom=False,top=False)
                    ax[dur_idx, tr_idx].spines["bottom"].set_visible(False)
                    ax[dur_idx, tr_idx].spines['right'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['top'].set_visible(False)
                    ax[dur_idx, tr_idx].spines['left'].set_visible(False)
                dur_idx+=1
            print(times[-1])
            dur=times[-1]
            if len(photometry[c][b][tr][times[-1]]) > 0:
                transients_start = np.zeros(7*seconds)
                transients_end = np.zeros(4 * seconds)
                for transient in photometry[c][b][tr][times[-1]]:
                    print(len(transient))
                    transients_start = np.vstack([transients_start,transient[:7*seconds]])
                    transients_end = np.vstack([transients_end,transient[-4*seconds:]])
                transients_start = transients_start[1:,:]
                transients_end = transients_end[1:, :]
                assert transients_start.shape[0]==transients_end.shape[0]
                num_of_transients = transients_start.shape[0]
                print(b, c, tr, dur, len(photometry[c][b][tr][dur]))
                plt.sca(ax[dur_idx, tr_idx])
                start_mean = np.nanmean(transients_start, axis=0)
                start_stderr = np.sqrt(np.nanvar(transients_start, axis=0) / num_of_transients)
                end_mean = np.nanmean(transients_end, axis=0)
                end_stderr = np.sqrt(np.nanvar(transients_end, axis=0) / num_of_transients)
                plt.plot(np.arange(7*seconds) , start_mean, label= '$(N=' + str(num_of_transients) + ')$', c=dur_colors[dur_idx], alpha=line_alpha)
                plt.fill_between(np.arange(7*seconds), y1=start_mean - start_stderr, y2=start_mean + start_stderr, color=c1, alpha=low_alpha)
                plt.plot(np.arange(11 * seconds,15*seconds,1), end_mean,  c=dur_colors[dur_idx],alpha=line_alpha)
                plt.fill_between(np.arange(11 * seconds,15*seconds,1), y1 = end_mean - end_stderr, y2=end_mean + end_stderr, color=c1, alpha=low_alpha)
                plt.legend()
                plt.xlim([0, 15 * seconds])
                plt.xticks(np.arange(0,(15 * seconds)+1,1*seconds ), [-2, -1, 0, 1, 2,3,4,5,None,None, None, -2, -1, 0, 1, 2])
                plt.ylabel('${\Delta F/F (Z)}$')
                plt.ylim(lims[sig_type][c])
                plt.title(labels[dur_idx])
                plt.vlines(x=2*seconds, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                plt.vlines(x=13 * seconds, ymin=lims[sig_type][c][0], ymax=lims[sig_type][c][1], ls='dashed', colors='k')
                colors = ["k", "white", "k"]
                x = [0, 7 * seconds, 11 * seconds, 15 * seconds]
                y = [0, 0, 0, 0]
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, colors=colors, linewidth=1, transform=ax[4, tr_idx].get_xaxis_transform(), clip_on=False)
                ax[4, tr_idx].add_collection(lc)
                ax[4, tr_idx].spines["bottom"].set_visible(False)
                ax[4, tr_idx].spines['right'].set_visible(False)
                ax[4, tr_idx].spines['top'].set_visible(False)
                xticks = ax[4, tr_idx].xaxis.get_major_ticks()
                for tick in range(8,11):
                    xticks[tick].set_visible(False)
            else:
                plt.sca(ax[4, tr_idx])
                plt.tick_params(left=False, right=False, labelleft=False,
                                labelbottom=False, bottom=False, top=False)
                ax[dur_idx, tr_idx].spines["bottom"].set_visible(False)
                ax[dur_idx, tr_idx].spines['right'].set_visible(False)
                ax[dur_idx, tr_idx].spines['top'].set_visible(False)
                ax[dur_idx, tr_idx].spines['left'].set_visible(False)
            dur_idx += 1

            all_events_mat = np.array(all_events[c][b][tr])
            events_length = np.array(all_events_length[c][b][tr])
            indices = np.argsort(events_length)
            ax[len(times), tr_idx] = sns.heatmap(all_events_mat[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[len(times), tr_idx],cbar=False)
            if tr_idx==0:
                ax[len(times),tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[len(times),tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, 9 * seconds,1*seconds),((np.arange(0, 9*seconds, 1 * seconds) - 2 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])

            tr_idx+=1

        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.suptitle(c+';  $\\theta$$_{short}^{long}$ = 2s ; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_'+sig_type+'.png',dpi=300)
        plt.close()

#%% Quantification of transitions for each behavior and day
output_folder = 'May24/Figures/Photometry/Z_over_bout/Transitions/'
num_of_pathways = 2
DRD1 = 0
A2A = 1
transitions = np.zeros((num_of_pathways,num_of_trials-2,num_of_behaviors,num_of_behaviors))
trials_groupedSaline = np.array(['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5','splashTest'])
print(np.argwhere(trials_groupedSaline=='saline'))
num_of_trials_groupedSaline = len(trials_groupedSaline)
for c in CMT:
    for m in CMT[c]:
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = DRD1
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = A2A
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = A2A
            else:
                pathway = DRD1
        else:
            break

        for tr in CMT[c][m]:
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                print(c, m, tr)
                predictions = CMT[c][m][tr]['merged']
                if 'saline' in tr: tr_idx = np.argwhere(trials_groupedSaline=='saline')
                else:
                    tr_idx = np.argwhere(trials_groupedSaline==tr)
                transitions[pathway, tr_idx, :, :] += hf.quantify_transitions(predictions, num_of_behaviors)




print('Done collecting data')
dir = ['Drd1','A2a']
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
behaviors_tag = ['GR','BL','WL','FL','RE','BC','ST','LM','JU']
for pathway in [DRD1,A2A]:
    fig,ax = plt.subplots(ncols=num_of_trials-2,figsize=(15,3))
    for tr_idx in range(num_of_trials_groupedSaline):
        plt.sca(ax[tr_idx])
        plt.imshow(transitions[pathway,tr_idx,:,:])
        plt.ylabel('${B_t}$')
        plt.xlabel('${B_{t+1}}$')
        plt.xticks(np.arange(num_of_behaviors),behaviors_tag,rotation=300)
        plt.yticks(np.arange(num_of_behaviors),behaviors_tag)
        plt.title(trials_groupedSaline[tr_idx])
    plt.suptitle('Transitions')
    plt.tight_layout()
    plt.savefig(output_folder+'/'+dir[pathway]+'.png')
    plt.close()

#%% Isolating per-transition-transients (PTTs) around transitions point
output_folder = 'May24/Figures/Photometry/Z_over_bout/Transitions/'
window = 90
num_of_pathways = 2
DRD1 = 0
A2A = 1
trials_groupedSaline = np.array(['saline','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5','splashTest'])
num_of_trials_groupedSaline = len(trials_groupedSaline)
PTT = {}
for pathway in range(num_of_pathways):
    PTT[pathway]={}
    for tr in trials_groupedSaline:
        PTT[pathway][tr]={}
        for b1 in range(num_of_behaviors):
            for b2 in range(num_of_behaviors):
                PTT[pathway][tr][(b1, b2)] = []

for c in CMT:
    for m in CMT[c]:
        if c in ['drd1_hm4di', 'drd1_hm3dq']:
            pathway = DRD1
        elif c in ['a2a_hm4di', 'a2a_hm3dq', 'a2a_opto']:
            pathway = A2A
        elif c == 'controls':
            if m.startswith('cA'):
                pathway = A2A
            else:
                pathway = DRD1
        else:
            break

        for tr in CMT[c][m]:
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                predictions = CMT[c][m][tr]['merged']
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['z']
                    right = CMT[c][m][tr]['photom']['right']['z']
                    photom = [left,right]
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['z']
                    photom = [left]
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['z']
                    photom = [right]
                print(c, m, tr)
                PTT_tr = hf.get_PTT(predictions,num_of_behaviors,photom)
                if 'saline' in tr: tr = 'saline'
                for pair in PTT_tr:

                    PTT[pathway][tr][pair].extend(PTT_tr[pair])
                    # if pair[1]==3: print(tr,pair , len(PTT[pathway][tr][pair]))
print('Done collecting data')
dir = ['Drd1','A2a']
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
for pathway in [DRD1,A2A]:
    for goTO_b in range(num_of_behaviors):
        fig,ax = plt.subplots(ncols=num_of_trials_groupedSaline,figsize=(18,3))
        tr_idx = 0
        for tr in trials_groupedSaline:
            plt.sca(ax[tr_idx])
            for b_prev in range(num_of_behaviors):
                if len(PTT[pathway][tr][(b_prev,goTO_b)])>0:
                    ptt = np.array(PTT[pathway][tr][(b_prev,goTO_b)])
                    mean = np.nanmean(ptt,axis=0)
                    stderr = np.sqrt(np.nanvar(ptt, axis=0)/ptt.shape[0])
                    plt.plot(np.arange(window),mean,c = colors[b_prev])
                    plt.fill_between(np.arange(window),mean+stderr,mean-stderr,color=colors[b_prev],alpha=.3)

            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0,window,FPS),(np.arange(0,window,FPS)//FPS).astype(int)-3)

            if tr_idx ==0 : plt.ylabel('${\Delta{F/F}(Z)}$')
            plt.vlines(x=window/2,ymin = plt.ylim()[1],ymax=plt.ylim()[0],linestyles='--',colors='k')
            plt.title(tr)
            tr_idx+=1

        plt.suptitle(dir[pathway]+';Per-Transition-Transient - GO TO behavior: '+behaviors[goTO_b])
        plt.tight_layout()
        plt.savefig(output_folder + dir[pathway] + '/'+behaviors[goTO_b]+'.png')
        plt.close()
#%% Heatmaps and means of behavior related transients (BRTs) from bout onset - z scored over bout - grouped short and long
sig_type = 'z_over_bout'
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Summary'
sig = {'left':[],'right':[]}
c2 = 'green'
c1 = 'red'
high_alpha = 0.4
low_alpha = 0.3
line_alpha = 0.8
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
lims = {'df':{'drd1':[-0.05,0.05],'a2a':[-0.01,0.01]},'z':{'drd1':[-.7,2],'a2a':[-.7,2]},'z_over_bout':{'drd1':[-.7,2],'a2a':[-.7,2]}}
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]=[]
            bout_length[pathway][b][tr]=[]
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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']
                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    bouts = hf.find_bouts(predictions, b)
                    for bout_idx in range(len(bouts)):
                        if 'saline' in tr: tr = 'saline'
                        if side== 'left':
                            if bouts[bout_idx][0] >= 5*seconds and left.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_left_tot = left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_left = np.mean(left[bouts[length][bout_idx][0] - 5*seconds:bouts[length][bout_idx][0] - 2*seconds])
                                z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                photometry[pathway][b][tr].append(z_over_bout_left)
                        elif side == 'right':
                            if bouts[bout_idx][0] >= 5*seconds and right.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_right_tot = right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_right = np.mean(right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                photometry[pathway][b][tr].append(z_over_bout_right)
                        else:
                            if bouts[bout_idx][0] >= 5*seconds and left.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_left_tot = left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_left = np.mean(left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))

                                sig_right_tot = right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_right = np.mean(right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))

                                bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                photometry[pathway][b][tr].append(z_over_bout_left)
                                photometry[pathway][b][tr].append(z_over_bout_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6,7]
for b in behaviors_prime:
    for c in cohorts:
        fig, ax = plt.subplots(nrows=2, ncols=7, sharex=True, figsize=(20,9))
        tr_idx = 0
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            print(b,c,tr)
            if 'saline' in tr :tr='saline'
            sig = np.array(photometry[c][b][tr])
            if sig.shape[0] ==0 :
                tr_idx+=1
                continue
            n_sig = sig.shape[0]
            mean = np.nanmean(sig,axis=0)
            stderr = np.sqrt(np.nanvar(sig, axis=0)/sig.shape[0])
            ax[0,tr_idx].plot(mean,c=c1, alpha=line_alpha)
            ax[0,tr_idx].fill_between(np.arange(window_length), y1=mean - stderr,y2=mean + stderr, color=c1, alpha=low_alpha)
            n_str = 'N = '+str(n_sig)
            all_events = np.copy(sig)
            if tr_idx==0:
                ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z)',fontsize=12)
            ax[0,tr_idx].set_ylim(lims[sig_type][c])
            ax[0,tr_idx].vlines(x=bout_onset, ymin=lims[sig_type][c][0],ymax=lims[sig_type][c][1], ls='dashed', colors='k')
            # ax[0,tr_idx].legend()
            plt.sca(ax[0,tr_idx])
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')
            events_length = np.array(bout_length[c][b][tr])
            indices = np.argsort(events_length)
            ax[1, tr_idx] = sns.heatmap(all_events[indices,], vmin= lims[sig_type][c][0], vmax=lims[sig_type][c][1], ax=ax[1, tr_idx],cbar=False)
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Event #',fontsize=12)
            plt.sca(ax[1,tr_idx])
            plt.xlabel('Time(s)')
            plt.xticks(np.arange(0, window_length + 1, 2 * seconds),
                       ((np.arange(0, window_length + 1, 2 * seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
            plt.yticks(np.linspace(0,events_length.shape[0],5,dtype=int),np.linspace(0,events_length.shape[0],5,dtype=int),fontsize=10)
            plt.sca(ax[0,tr_idx])
            plt.title(tr)
            tr_idx+=1
        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.suptitle(c+'; Quiet BL period = 3s ;  Behavior : '+behaviors[b])
        plt.savefig(output_folder +'/Onset/'+ c + '/'+behaviors[b]+'_heatmapsAndMeans_groupedLongShort_'+sig_type+'.png',dpi=300)
        plt.close()





#%% Heatmaps and means of behavior related transients (BRTs) from bout onset - z scored over bout - grouped short and long - A2a & Drd1 on the same plot
sig_type = 'z_over_bout'
output_folder = root_folder+'Figures/Photometry/'+sig_type+'/Summary'
sig = {'left':[],'right':[]}
c2 = '#f8acff'
c1 = '#696eff'
high_alpha = 0.7
low_alpha = 0.5
line_alpha = 1
y_lims = [-.7,1.2]
theta = 2
lw=1
photometry = {}
bout_length = {}
seconds = 15
window_length = 15*seconds
bout_onset = 5*seconds
for pathway in ['drd1', 'a2a']:
    photometry[pathway] = {}
    bout_length[pathway] = {}
    for b in np.arange(num_of_behaviors):
        photometry[pathway][b] = {}
        bout_length[pathway][b] = {}
        for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
            photometry[pathway][b][tr]=[]
            bout_length[pathway][b][tr]=[]
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
            if tr not in trials: continue
            if 'photom' in CMT[c][m][tr].keys():
                if 'left' in CMT[c][m][tr]['photom'] and 'right' in CMT[c][m][tr]['photom'] :
                    side = 'both'
                    left = CMT[c][m][tr]['photom']['left']['df']
                    right = CMT[c][m][tr]['photom']['right']['df']
                elif 'left' in CMT[c][m][tr]['photom'] and 'right' not in CMT[c][m][tr]['photom'] :
                    side = 'left'
                    left = CMT[c][m][tr]['photom']['left']['df']
                else:
                    side = 'right'
                    right = CMT[c][m][tr]['photom']['right']['df']
                predictions = CMT[c][m][tr]['merged']
                for b in np.arange(num_of_behaviors):
                    print(c , m , tr , behaviors[b])
                    bouts = hf.find_bouts(predictions, b)
                    for bout_idx in range(len(bouts)):
                        if 'saline' in tr: tr = 'saline'
                        if side== 'left':
                            if bouts[bout_idx][0] >= 5*seconds and left.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_left_tot = left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_left = np.mean(left[bouts[length][bout_idx][0] - 5*seconds:bouts[length][bout_idx][0] - 2*seconds])
                                z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))
                                if not np.all(np.isnan(z_over_bout_left)):
                                    bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                    photometry[pathway][b][tr].append(z_over_bout_left)
                        elif side == 'right':
                            if bouts[bout_idx][0] >= 5*seconds and right.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_right_tot = right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_right = np.mean(right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))
                                if not np.all(np.isnan(z_over_bout_right)):
                                    bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                    photometry[pathway][b][tr].append(z_over_bout_right)
                        else:
                            if bouts[bout_idx][0] >= 5*seconds and left.shape[0] >= bouts[bout_idx][0] + 10*seconds:
                                sig_left_tot = left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_left = np.mean(left[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_left = stats.zscore(stats.zscore(sig_left_tot,nan_policy='omit'))

                                sig_right_tot = right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] + 10*seconds]
                                # baseline_right = np.mean(right[bouts[bout_idx][0] - 5*seconds:bouts[bout_idx][0] - 2*seconds])
                                z_over_bout_right = stats.zscore(stats.zscore(sig_right_tot,nan_policy='omit'))

                                if not np.all(np.isnan(z_over_bout_left)):
                                    bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                    photometry[pathway][b][tr].append(z_over_bout_left)
                                if not np.all(np.isnan(z_over_bout_right)):
                                    bout_length[pathway][b][tr].append(len(bouts[bout_idx]))
                                    photometry[pathway][b][tr].append(z_over_bout_right)

cohorts = ['drd1','a2a']
behaviors_prime = [0,1,2,3,4,6,7]
for b in behaviors_prime:
    fig, ax = plt.subplots(nrows=3, ncols=7, sharex=True, figsize=(20, 9))
    tr_idx = 0
    for tr in ['saline', 'cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','splashTest']:
        drd1_sig = np.array(photometry['drd1'][b][tr])
        a2a_sig = np.array(photometry['a2a'][b][tr])
        if 'saline' in tr :tr='saline'
        if drd1_sig.shape[0] ==0 and a2a_sig.shape[0]==0:
            tr_idx+=1
            continue
        drd1_n = drd1_sig.shape[0]
        a2a_n = a2a_sig.shape[0]
        print(tr , drd1_n,a2a_n)
        drd1_mean = np.nanmean(drd1_sig,axis=0)
        drd1_stderr = np.sqrt(np.nanvar(drd1_sig, axis=0)/drd1_sig.shape[0])

        a2a_mean = np.nanmean(a2a_sig, axis=0)
        a2a_stderr = np.sqrt(np.nanvar(a2a_sig, axis=0) / a2a_sig.shape[0])

        plt.sca(ax[0,tr_idx])
        plt.plot(drd1_mean,c=c1, alpha=line_alpha, label = 'Drd1')
        plt.fill_between(np.arange(window_length), y1=drd1_mean - drd1_stderr,y2=drd1_mean + drd1_stderr, color=c1, alpha=low_alpha)

        plt.plot(a2a_mean, c=c2, alpha=line_alpha, label='A2a')
        plt.fill_between(np.arange(window_length), y1=a2a_mean - a2a_stderr, y2=a2a_mean + a2a_stderr, color=c2,alpha=low_alpha)
        if tr_idx==0:
            ax[0,tr_idx].set_ylabel('Mean $\\Delta$F/F (Z)',fontsize=12)
        ax[0,tr_idx].set_ylim(y_lims)
        ax[0,tr_idx].vlines(x=bout_onset, ymin=y_lims[0],ymax=y_lims[1], ls='dashed', colors='k')
        ax[0,tr_idx].legend()
        plt.sca(ax[0,tr_idx])
        plt.xticks(np.arange(0, window_length + 1, 2 * seconds), '')

        drd1_all_events = np.copy(drd1_sig)
        a2a_all_events = np.copy(a2a_sig)

        drd1_indices = np.argsort(np.array(bout_length['drd1'][b][tr]))
        a2a_indices = np.argsort(np.array(bout_length['a2a'][b][tr]))
        if drd1_n>0:
            ax[1, tr_idx] = sns.heatmap(drd1_all_events[drd1_indices,], vmin= y_lims[0], vmax=y_lims[1], ax=ax[1, tr_idx],cbar=False,cmap = sns.color_palette("light:b", as_cmap=True))
            if tr_idx==0:
                ax[1,tr_idx].set_ylabel('Drd1 event #',fontsize=12)
        if a2a_n>0:
            ax[2, tr_idx] = sns.heatmap(a2a_all_events[a2a_indices,], vmin=y_lims[0], vmax=y_lims[1],ax=ax[2, tr_idx], cbar=False,cmap = sns.light_palette("purple", as_cmap=True))
            if tr_idx == 0:
                ax[2, tr_idx].set_ylabel('A2a event #', fontsize=12)
        plt.sca(ax[1,tr_idx])
        plt.yticks(np.linspace(0, drd1_all_events.shape[0], 5, dtype=int),np.linspace(0, drd1_all_events.shape[0], 5, dtype=int), fontsize=10)
        plt.sca(ax[2,tr_idx])
        plt.xlabel('Time(s)')
        plt.xticks(np.arange(0, window_length + 1, 2 * seconds),((np.arange(0, window_length + 1, 2 * seconds) - 5 * seconds) / seconds).astype(int), rotation='horizontal', fontsize=10)
        plt.yticks(np.linspace(0,a2a_all_events.shape[0],5,dtype=int),np.linspace(0,a2a_all_events.shape[0],5,dtype=int),fontsize=10)
        plt.sca(ax[0,tr_idx])
        plt.title(tr)
        tr_idx+=1
    fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
    plt.suptitle('Drd1 & A2a Transients Aligned To Bout Onset\n Quiet BL period = 3s ;  Behavior : '+behaviors[b])
    plt.savefig(output_folder +'/Onset/Both_pathways/'+behaviors[b]+'_heatmapsAndMeans_groupedLongShort_'+sig_type+'.png',dpi=300)
    plt.close()


