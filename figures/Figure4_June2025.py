import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import bz2
import seaborn as sns
import matplotlib.pyplot as plt
import helper_functions as hf
import scipy.signal as signal
import pickle
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CTM_Dec24.pkl', 'rb')
stim_days = pickle.load(ifile)['a2a_opto']
stim_days = {key:stim_days[key] for key in ['cocaine6laserStim','cocaine8laserStim']}
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
SSD_color = '#CC6677'
velocity_color = '#44AA99'
FPS=15
sample_rate=15
second = 15
minute = 60*second
ISI = 1 * minute + 20*second
PRE = 0
DURING = 1
POST = 2
bin_duration = 20 * second
num_of_bins = 3
num_of_behaviors = len(behaviors)
num_of_stims = 14
num_of_mice = len(stim_days['cocaine6laserStim'].keys())
RNN_offset = 7
PATHO_LICKING = 0
NATURAL_LICKING = 1
NO_LICKING = 2
mm=1/25.4
grouping_lut =  np.array([NO_LICKING,NO_LICKING,PATHO_LICKING,PATHO_LICKING,NATURAL_LICKING,NATURAL_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
#%% Fig 4E
output_folder = root_folder+'/Figures/Optogenetics/A2a/Ethograms/'
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
for t in stim_days.keys():
    for mouse in stim_days[t]:
        print(mouse)
        predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']
        velocity = stim_days[t][mouse]['topcam']['velocity'] * FPS * 10
        SSD = hf.smoothing(np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0),
                           3) * 1000
        stims = np.empty_like(predictions)
        stims[:] = 11
        for stim in stim_times:
            stims[stim:stim + 20 * second] = 10
        predictions = np.reshape(predictions, (1, predictions.shape[0]))
        stims = np.reshape(stims, (1, stims.shape[0]))
        fig,ax = plt.subplots(nrows=4,sharex='all',figsize=(100*mm, 45*mm),gridspec_kw={'height_ratios': [1, 4,3,3]})
        ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,cbar=False,vmin=0, vmax=num_of_behaviors-1, ax=ax[1])
        plt.sca(ax[1])
        ax[0] = sns.heatmap(stims, yticklabels=[''],cbar = False, cmap=['#B2E2F6','white'], vmin=num_of_behaviors+1, vmax=num_of_behaviors+2,ax= ax[0])
        ax[0].tick_params(left=False, bottom=False)
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        plt.sca(ax[0])
        ax[2].plot(velocity,lw=.3)
        ax[2].bar(x=np.nonzero(stims == 10)[1], height=np.max(velocity), width=1, color='#B2E2F6' )
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        plt.sca(ax[2])
        # plt.ylabel('Velocity\n(m/s)')
        ax[3].plot(SSD, lw=.3, color='gray')
        ax[3].bar(x=np.nonzero(stims == 10)[1], height=np.nanmax(SSD), width=1, color='#B2E2F6')
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        plt.sca(ax[3])
        plt.xticks(np.arange(0, predictions.shape[1], 900), (np.arange(0, predictions.shape[1], 900) / 900).astype(int),rotation=0, fontsize=8)
        fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
        plt.savefig(output_folder+'/'+mouse+'_'+t+'.png',dpi=300)
        plt.savefig(output_folder + '/' + mouse + '_' + t + '.pdf', dpi=300)
        plt.close()
#%% Fig 4F
output_folder = 'output/Optogenetics/A2a/Summary/'
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
laser_status = np.zeros(28 * minute)
smooth_window_len = 60
smooth_window = np.ones(smooth_window_len) / smooth_window_len
m_idx=0
for mouse in stim_days['cocaine6laserStim']:
    # print(mouse)
    if np.count_nonzero((stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge'][:first_stim]==2) | (stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge'][:first_stim]==3))/first_stim>=0.5:
        print(6,mouse)
        t='cocaine6laserStim'
    elif np.count_nonzero((stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge'][:first_stim]==2) | (stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge'][:first_stim]==3))/first_stim>=0.5:
        print(8,mouse)
        t = 'cocaine8laserStim'
    else:
        print('None',mouse)
        continue
    m_predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']
    if m_idx==0:
        predictions = np.reshape(np.copy(m_predictions),(1,m_predictions.size))
        print(predictions.shape)
    else:
        delta = predictions.shape[1]-m_predictions.size
        if delta>0:
            predictions=predictions[:,:-delta]
        elif delta<0:
            m_predictions = m_predictions[:delta]
        print(predictions.shape,m_predictions.shape)
        predictions = np.vstack([predictions,m_predictions])
    m_idx+=1
print(predictions.shape)
dist = np.zeros((num_of_behaviors,predictions.shape[1]))
for idx in range(predictions.shape[1]):
    for b in range(num_of_behaviors):
        dist[b,idx]=np.count_nonzero(predictions[:,idx]==b)/predictions.shape[0]

fig,ax = plt.subplots(nrows=2,sharex='all',gridspec_kw={'height_ratios': [1, 20]},figsize=(102*mm,47*mm))
time = np.arange(predictions.shape[1])
for row in range(dist.shape[0]):
    dist[row,:]= signal.filtfilt(smooth_window, 1, dist[row,:])
plt.sca(ax[1])
plt.stackplot(time,dist[0],dist[1],dist[2],dist[3],dist[4],dist[5],dist[6],dist[7],dist[8],labels = behaviors,colors=colors)
plt.xticks(np.arange(0,predictions.shape[1],4500),np.arange(0,30,5),fontsize=8)
plt.xlabel('Time(m)',fontsize=10)
plt.ylabel('Probability',fontsize=10)
# plt.legend(frameon=False,fontsize=8)
plt.xlim(0,predictions.shape[1])
plt.ylim(0,1)
plt.yticks([0,0.5,1],[0,0.5,1],fontsize=8)
plt.gca().tick_params(left=True, bottom=True)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.sca(ax[0])
plt.bar(stim_times,height=1,width=bin_duration,color='#B2E2F7',align='edge')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().tick_params(left=False, bottom=False,labelleft=False)
plt.sca(ax[1])
plt.vlines(x = stim_times,ymin=0,ymax=1,colors='k',linestyles='--',lw=.5)
plt.vlines(x = stim_times+bin_duration,ymin=0,ymax=1,colors='k',linestyles='--',lw=.5)
fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
plt.savefig(output_folder+'stackplot.png',dpi=300)
plt.savefig(output_folder+'stackplot.pdf',dpi=300)
plt.close()
#%% Fig 4G,I
output_folder = root_folder+'/Figures/Optogenetics/A2a/Summary/'
stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,num_of_behaviors),dtype=np.float32)
clustered_stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,3),dtype=np.float32)
m_idx = 0
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    cocaine6_grouped = grouping_lut[stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge']]
    cocaine8_grouped = grouping_lut[stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge']]
    if np.count_nonzero(cocaine6_grouped[:first_stim]==PATHO_LICKING)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(cocaine8_grouped[:first_stim]==PATHO_LICKING)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        # m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']
    velocity = stim_days[t][mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            stim_epoch_behavior[PRE, stim_idx, m_idx, behavior]= np.count_nonzero(predictions[laser_onset-bin_duration:laser_onset]==behavior)/bin_duration
            stim_epoch_behavior[DURING, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset: laser_onset + bin_duration] == behavior)/bin_duration
            stim_epoch_behavior[POST, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset + bin_duration:laser_onset+ (2*bin_duration)] == behavior)/bin_duration
        clustered_predictions = np.copy(predictions)
        clustered_predictions = grouping_lut[clustered_predictions]
        for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
            clustered_stim_epoch_behavior[PRE, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset - bin_duration:laser_onset] == category)/bin_duration
            clustered_stim_epoch_behavior[DURING, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset: laser_onset + bin_duration] == category)/bin_duration
            clustered_stim_epoch_behavior[POST, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset + bin_duration:laser_onset + (2 * bin_duration)] == category)/bin_duration
    m_idx+=1
# num_of_mice_wo_outliers = num_of_mice-len(outliers)
stim_epoch_behavior = stim_epoch_behavior[:,:,:m_idx,:]
clustered_stim_epoch_behavior = clustered_stim_epoch_behavior[:,:,:m_idx,:]
titles = ['Surface licking','Self licking','No licking']
grouped_colors = ["#d73027", "#c4a7e7", "#1f4e79"]
for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
    plt.figure(figsize=(2,1.8),frameon=False)
    plt.plot(np.mean(clustered_stim_epoch_behavior[:,:,:,category],axis=1),alpha=0.3,marker='o',markersize=2,lw=1,color='gray')
    plt.bar(1,height=1 , color='#009FE3', alpha=.3)
    plt.plot(np.mean(clustered_stim_epoch_behavior[:,:,:,category],axis=(1,2)),color=grouped_colors[category])
    plt.errorbar(np.arange(3),y=np.mean(clustered_stim_epoch_behavior[:,:,:,category],axis=(1,2)),
                 yerr=np.sqrt(np.var(clustered_stim_epoch_behavior[:,:,:,category])/(clustered_stim_epoch_behavior.shape[1]*clustered_stim_epoch_behavior.shape[2])).T,
                 color=grouped_colors[category],capsize=2,capthick=1,lw=1)
    plt.xticks(np.arange(3),['Prior','During','Post'],fontsize=8)
    plt.ylabel('% Time spent',fontsize=10)
    plt.yticks([0,0.5,1],[0,50,100],fontsize=8)
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    # plt.title(titles[category])
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder + '/V_graph_'+titles[category]+'.png', dpi=300)
    plt.savefig(output_folder + '/V_graph_' + titles[category] + '.pdf', dpi=300)
    plt.close()
print('Outliers : ',outliers)
#%% Fig 4H,J
output_folder = root_folder+'/Figures/Optogenetics/A2a/Dynamics/'
dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
clustered_dynamics = np.zeros((num_of_mice,num_of_stims,3,minute),dtype=np.float32)
m_idx = 0
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
outliers = []

for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    cocaine6_grouped = grouping_lut[stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge']]
    cocaine8_grouped = grouping_lut[stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge']]
    if np.count_nonzero(cocaine6_grouped[:first_stim]==PATHO_LICKING)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(cocaine8_grouped[:first_stim]==PATHO_LICKING)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        # m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']
    velocity = stim_days[t][mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            dynamics[m_idx , stim_idx, behavior,:] = predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==behavior
        clustered_predictions = np.copy(predictions)
        clustered_predictions = grouping_lut[clustered_predictions]
        for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
            clustered_dynamics[m_idx , stim_idx, category,:] = clustered_predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==category
    m_idx+=1
# num_of_mice_wo_outliers = num_of_mice-len(outliers)
dynamics = dynamics[:m_idx,:,:,:]
clustered_dynamics = clustered_dynamics[:m_idx,:,:,:]
titles = ['Surface licking','Self licking','No licking']
grouped_colors = ["#d73027", "#c4a7e7", "#1f4e79"]
for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
    psth = np.mean(clustered_dynamics[:,:,category,:],axis=1)
    for m in range(m_idx):
        psth[m] = hf.smoothing(psth[m],5)
    mean_psth = np.mean(psth,axis=0)
    stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
    plt.figure(figsize=(2.2,2.2),frameon=False)
    plt.plot(mean_psth,color=grouped_colors[category],alpha=.7,lw=1)
    plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color=grouped_colors[category],alpha=.3)
    plt.xticks(np.arange(0, minute, 10 * second), (np.arange(-bin_duration, 2 * bin_duration, 10 * second) / second).astype(int),fontsize=8)
    plt.bar(x=bin_duration,width=bin_duration,height=1,color='#009FE3', alpha=.3,align='edge')
    plt.ylabel('Probability',fontsize=10)
    plt.xlabel('Time(s)',fontsize=10)
    plt.yticks([0,0.5,1],[0,0.5,1],fontsize=8)
    plt.ylim([0,1])
    # plt.title(titles[category])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder + '/dynamics_'+titles[category].replace(' ','_')+'.png', dpi=300)
    plt.savefig(output_folder + '/dynamics_' + titles[category].replace(' ', '_') + '.pdf', dpi=300)
    plt.close()
#%% Fig 4K,M
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
output_folder = root_folder+'/Figures/Optogenetics/A2a/Movement_parameters/'
num_of_cams=2
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
m_idx = 0
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    cocaine6_grouped = grouping_lut[stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge']]
    cocaine8_grouped = grouping_lut[stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge']]
    if np.count_nonzero(cocaine6_grouped[:first_stim] == PATHO_LICKING) / first_stim >= 0.5:
        t = 'cocaine6laserStim'
    elif np.count_nonzero(cocaine8_grouped[:first_stim] == PATHO_LICKING) / first_stim >= 0.5:
        t = 'cocaine8laserStim'
    else:
        # m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']

    laser_times=[]
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        laser_times.extend(np.arange(laser_onset,laser_onset+bin_duration,1))
    for behavior in range(num_of_behaviors):
        laser_on[behavior].append([])
        laser_off[behavior].append([])
        bout = []
        for t in range(predictions.size):
            if predictions[t] == behavior:
                bout.append(t)
            else:
                if len(bout)==0:continue
                if np.intersect1d(np.array(bout),laser_times).size>len(bout)/2:
                    laser_on[behavior][-1].append(len(bout))
                else:

                    laser_off[behavior][-1].append(len(bout))
                bout=[]
    m_idx+=1
lims_idx=0
for behavior in [2,3,4,5,7]:
    plt.figure(figsize=(50*mm,50*mm), frameon=False)
    meanOfMeans_on = []
    meanOfMeans_off = []
    for m in laser_on[behavior]:
        if len(m)==0:continue
        mean = np.mean(m)
        meanOfMeans_on.append(mean)
    for m in laser_off[behavior]:
        if len(m) == 0: continue
        mean = np.mean(m)
        meanOfMeans_off.append(mean)
    meanOfMeans_on= np.mean(meanOfMeans_on)
    meanOfMeans_off = np.mean(meanOfMeans_off)
    plt.bar([0,.5],height=[meanOfMeans_on,meanOfMeans_off],color='white',edgecolor=colors[behavior],width=.3)
    for m in laser_on[behavior]:
        if len(m)==0:continue
        mean = np.mean(m)
        err = np.sqrt(np.var(m)/len(m))
        plt.errorbar(0, mean, yerr=err, capsize=2, capthick=1, c=colors[behavior])
        plt.scatter(0,mean,c='white',edgecolors=colors[behavior])
    for m in laser_off[behavior]:
        if len(m) == 0: continue
        mean = np.mean(m)
        err = np.sqrt(np.var(m)/len(m))
        plt.errorbar(.5, mean, yerr=err, capsize=2, capthick=1, c=colors[behavior])
        plt.scatter(.5, mean, c='white', edgecolors=colors[behavior])
    plt.ylabel('Mean bout \nduration (s)',fontsize=10)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks([0,.5],['On','Off'],fontsize=8)
    if behavior==2: plt.yticks([0,45,90],[0,3,6],fontsize=8)
    if behavior == 7: plt.yticks([0, 15, 30], [0, 1, 2 ], fontsize=8)
    plt.xlabel('Laser', fontsize=10)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
    plt.savefig(output_folder+'boutDur_laser_on_off_'+behaviors[behavior]+'.png', dpi=300)
    plt.savefig(output_folder + 'boutDur_laser_on_off_' + behaviors[behavior] + '.pdf', dpi=300)
    plt.close()
#%% Fig 4L,N
mm=1/25.4
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
output_folder = root_folder+'/Figures/Optogenetics/A2a/Movement_parameters/'
num_of_cams=2
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
m_idx = 0
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    cocaine6_grouped = grouping_lut[stim_days['cocaine6laserStim'][mouse]['merged']['predictions']['smartMerge']]
    cocaine8_grouped = grouping_lut[stim_days['cocaine8laserStim'][mouse]['merged']['predictions']['smartMerge']]
    if np.count_nonzero(cocaine6_grouped[:first_stim] == PATHO_LICKING) / first_stim >= 0.5:
        t = 'cocaine6laserStim'
    elif np.count_nonzero(cocaine8_grouped[:first_stim] == PATHO_LICKING) / first_stim >= 0.5:
        t = 'cocaine8laserStim'
    else:
        # m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']['predictions']['smartMerge']
    velocity = stim_days[t][mouse]['topcam']['velocity']*FPS*10
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)*1000
    laser_times = []
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        laser_times.extend(np.arange(laser_onset,laser_onset+bin_duration,1))
    for behavior in range(num_of_behaviors):
        behavior_indices = np.nonzero(predictions==behavior)[0]
        for idx in behavior_indices:
            if not np.isnan(SSD[idx]):
                if idx in laser_times:
                  laser_on[behavior].append([SSD[idx],velocity[idx]])
                else:
                    laser_off[behavior].append([SSD[idx],velocity[idx]])
    m_idx+=1
lims_idx=0
for behavior in [2,3,4,5,7]:
    laser_off[behavior] = np.array(laser_off[behavior])
    laser_on[behavior] = np.array(laser_on[behavior])
    plt.figure(figsize=(50*mm,50*mm), frameon=False)
    sns.histplot(laser_off[behavior][:,1],bins=50,color='gray',alpha=.5,label='Laser off',stat='probability')
    sns.histplot(laser_on[behavior][:,1], bins=50, color=colors[behavior], alpha=.5, label='Laser on',stat='probability')
    plt.xlabel('Velocity(cm/s)',fontsize=10)
    plt.legend(fontsize=8,frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(fontsize=8)
    if behavior in [2,7]:
        plt.yticks([0,0.075,0.15],[0,0.075,0.15],fontsize=8)


    lims_idx+=1
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
    plt.savefig(output_folder+'vel_laser_on_off_'+behaviors[behavior]+'_vel.png', dpi=300)
    plt.savefig(output_folder + 'vel_laser_on_off_' + behaviors[behavior] + '_vel.pdf', dpi=300)
    plt.close()
