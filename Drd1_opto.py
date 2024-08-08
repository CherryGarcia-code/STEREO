#%%
import numpy as np
import bz2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import helper_functions as hf
import scipy.signal as signal
import pickle
import scipy.stats as stats
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CTM_May24.pkl', 'rb')
stim_day = pickle.load(ifile)['drd1_opto']['AloneStim']
ifile.close()
temp_file = open(folder+'opto_alignment_dict.pkl', 'rb')
opto_dict = pickle.load(temp_file)
temp_file.close()
print('***All dictionaries were loaded***')
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163','#969595', '#b4a7d6']
FPS=15
sample_rate=15
second = 15
minute = 60*second
ISI = 1 * minute + 20*second
mpl.rcParams['xtick.labelsize']=10
mpl.rcParams['ytick.labelsize']=10
mpl.rcParams['legend.fontsize']=8
PRE = 0
DURING = 1
POST = 2
VDB = 0
nVDB = 1
bin_duration = 20 * second
num_of_bins = 3
num_of_behaviors = len(behaviors)
num_of_stims = 9
num_of_mice = len(stim_day.keys())
#%% Plot ethograms with stimulation ticks
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Ethograms/'
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'],stim_day[mouse]['SSD']['cam2']]),axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    stims = np.empty_like(predictions)
    stims[:]=10
    for stim in stim_times:
        stims[stim:stim+20*second]=9
    predictions = np.reshape(predictions, (1, predictions.shape[0]))
    stims = np.reshape(stims, (1, stims.shape[0]))
    fig,ax = plt.subplots(nrows=4,sharex='all',figsize=(24, 8),gridspec_kw={'height_ratios': [1, 4,3,3]})
    cbar_ax = fig.add_axes([.91, .3, .01, .4])
    ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,
                        cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5, 'drawedges': True},cbar_ax=cbar_ax,vmin=0, vmax=num_of_behaviors-1, ax=ax[1])
    cbar = ax[1].collections[0].colorbar
    cbar.set_ticklabels(behaviors)
    plt.sca(ax[1])
    plt.ylabel('Behavior')
    ax[0] = sns.heatmap(stims, yticklabels=[''],cbar = False, cmap=['#B2E2F6','white'], vmin=9, vmax=10,ax= ax[0])
    ax[0].tick_params(left=False, bottom=False)
    ax[0].spines['left'].set_visible(True)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(True)
    ax[0].spines['right'].set_visible(True)
    plt.sca(ax[0])
    plt.ylabel('Stim')
    ax[2].plot(velocity * FPS,c = '#44AA99')
    ax[2].bar(x=np.nonzero(stims == 9)[1], height=np.max(velocity)*FPS, width=1, color='#009FE3', alpha=.3)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    plt.sca(ax[2])
    plt.ylabel('Velocity\n(m/s)')
    ax[3].plot(SSD,color = '#CC6677')
    ax[3].bar(x=np.nonzero(stims==9)[1],height = np.max(SSD[~np.isnan(SSD)]),width = 1,color='#009FE3', alpha=.3)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    plt.sca(ax[3])
    plt.ylabel('SSD\n(a.u.)')
    plt.xticks(np.arange(0, predictions.shape[1], 900), (np.arange(offset, predictions.shape[1]+offset , 900) / 900).astype(int),rotation=0, fontsize=8)
    plt.xlabel('Time (minutes)')
    plt.suptitle('Drd1_opto , '+mouse+' , AloneStim')
    fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
    plt.savefig(output_folder+'/'+mouse+'.png',dpi=300)
    plt.close()
#%% V graph - VDB
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Summary/'
stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,num_of_behaviors),dtype=np.float32)
clustered_stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,2),dtype=np.float32)
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            stim_epoch_behavior[PRE, stim_idx, m_idx, behavior]= np.count_nonzero(predictions[laser_onset-bin_duration:laser_onset]==behavior)/bin_duration
            stim_epoch_behavior[DURING, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset: laser_onset + bin_duration] == behavior)/bin_duration
            stim_epoch_behavior[POST, stim_idx, m_idx, behavior] = np.count_nonzero(predictions[laser_onset + bin_duration:laser_onset+ (2*bin_duration)] == behavior)/bin_duration
        clustered_predictions = np.copy(predictions)
        clustered_predictions = np.where(clustered_predictions<=3,VDB,nVDB)
        for category in [VDB,nVDB]:
            clustered_stim_epoch_behavior[PRE, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset - bin_duration:laser_onset] == category)/bin_duration
            clustered_stim_epoch_behavior[DURING, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset: laser_onset + bin_duration] == category)/bin_duration
            clustered_stim_epoch_behavior[POST, stim_idx, m_idx,category] = np.count_nonzero(clustered_predictions[laser_onset + bin_duration:laser_onset + (2 * bin_duration)] == category)/bin_duration
    m_idx+=1
plt.figure()
plt.plot(np.mean(clustered_stim_epoch_behavior[:,:,:,VDB],axis=1),alpha=0.3,color='gray',marker='o')
plt.bar(1,height=1 , color='#009FE3', alpha=.3)
plt.plot(np.mean(clustered_stim_epoch_behavior[:,:,:,VDB],axis=(1,2)),color='#730AFF')
plt.errorbar(np.arange(3),y=np.mean(clustered_stim_epoch_behavior[:,:,:,VDB],axis=(1,2)),
             yerr=np.sqrt(np.var(clustered_stim_epoch_behavior[:,:,:,VDB])/(clustered_stim_epoch_behavior.shape[1]*clustered_stim_epoch_behavior.shape[2])).T,
             color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Prior','During','Post'])
plt.ylabel('Fraction of frames')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Change in behavior around laser epoch\n Behaviors: Grooming, body licking,wall licking,floor licking')
plt.savefig(output_folder + '/V_graph_VDB.png', dpi=300)
plt.close()

#%% V graphs - all behaviors
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Summary/'
for behavior in range(num_of_behaviors):
    plt.figure()
    plt.plot(np.mean(stim_epoch_behavior[:,:,:,behavior],axis=1),alpha=0.3,color='gray',marker='o')
    plt.bar(1,height=1 , color='#009FE3', alpha=.3)
    plt.plot(np.mean(stim_epoch_behavior[:,:,:,behavior],axis=(1,2)),color=colors[behavior])
    plt.errorbar(np.arange(3),y=np.mean(stim_epoch_behavior[:,:,:,behavior],axis=(1,2)),
                 yerr=np.sqrt(np.var(stim_epoch_behavior[:,:,:,behavior])/(stim_epoch_behavior.shape[1]*stim_epoch_behavior.shape[2])).T,
                 color=colors[behavior],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Prior','During','Post'])
    plt.ylabel('Fraction of frames')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Change in behavior around laser epoch\n Behaviors: '+behaviors[behavior])
    plt.savefig(output_folder + '/V_graph_'+behaviors[behavior]+ '.png', dpi=300)
    plt.close()

#%% Dynamics - VDB
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Dynamics/'
dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
clustered_dynamics = np.zeros((num_of_mice,num_of_stims,2,minute),dtype=np.float32)
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            dynamics[m_idx , stim_idx, behavior,:] = predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==behavior
        clustered_predictions = np.copy(predictions)
        clustered_predictions = np.where(clustered_predictions<=3,VDB,nVDB)
        for category in [VDB,nVDB]:
            clustered_dynamics[m_idx , stim_idx, category,:] = clustered_predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==category
    m_idx+=1

psth = np.mean(clustered_dynamics[:,:,VDB,:],axis=1)
for m in range(num_of_mice):
    psth[m] = hf.smoothing(psth[m],5)
mean_psth = np.mean(psth,axis=0)
stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
plt.figure()
plt.plot(mean_psth,color='#730AFF')
plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color='#730AFF',alpha=.3)
plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
plt.bar(x=np.arange(bin_duration,2*bin_duration,1),height=np.ones(bin_duration) ,width=1, color='#009FE3', alpha=.3)
plt.ylabel('Fraction of mice')
plt.xlabel('Time[s]')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
plt.title('Dynamics of Change in behavior around laser epoch\n Behaviors: Grooming, body licking,wall licking,floor licking')
plt.savefig(output_folder + '/dynamics_VDB.png', dpi=300)
plt.close()
#%% Dynamics- all behaviors
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Dynamics/'
for behavior in range(num_of_behaviors):
    psth = np.mean(dynamics[:,:,behavior,:],axis=1)
    for m in range(num_of_mice):
        psth[m] = hf.smoothing(psth[m],5)
    mean_psth = np.mean(psth,axis=0)
    stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
    plt.figure()
    plt.plot(mean_psth,color=colors[behavior])
    plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color=colors[behavior],alpha=.3)
    plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
    plt.bar(x=np.arange(bin_duration, 2 * bin_duration, 1), height=np.ones(bin_duration), width=1, color='#009FE3', alpha=.3)
    plt.ylabel('Fraction of mice')
    plt.xlabel('Time[s]')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    plt.title('Dynamics of Change in behavior around laser epoch\n Behaviors: '+behaviors[behavior])
    plt.savefig(output_folder + '/dynamics_'+behaviors[behavior]  + '.png', dpi=300)
    plt.close()

#%% Velocity & vigor around laser onset
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Movement_parameters/'
num_of_cams=2
velocity_matrix = np.zeros((num_of_mice,num_of_stims,minute),dtype=np.float32)
SSD_matrix = np.zeros((num_of_mice,num_of_stims,minute),dtype=np.float32)
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        velocity_matrix[m_idx,stim_idx,:]=velocity[laser_onset-bin_duration:laser_onset+(2*bin_duration)]
        SSD_matrix[m_idx,stim_idx,:]=SSD[laser_onset-bin_duration:laser_onset+(2*bin_duration)]

    m_idx+=1

velocity_matrix *=FPS
vel_mouse_mean = np.mean(velocity_matrix[:,:,:],axis=1)
SSD_mouse_mean = np.nanmean(SSD_matrix[:,:,:],axis=1)
vel_overall_mean = np.mean(vel_mouse_mean,axis=0)
vel_overall_stderr = np.sqrt(np.var(vel_mouse_mean,axis=0)/vel_mouse_mean.shape[0])
SSD_overall_mean = np.nanmean(SSD_mouse_mean,axis=0)
SSD_overall_stderr = np.sqrt(np.nanvar(SSD_mouse_mean,axis=0)/SSD_mouse_mean.shape[0])


plt.figure()
plt.plot(vel_overall_mean,color='#44AA99')
plt.fill_between(np.arange(minute),y1 = vel_overall_mean-vel_overall_stderr,y2=vel_overall_mean+vel_overall_stderr,color='#44AA99',alpha=.3)
plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
plt.bar(x=np.arange(bin_duration, 2 * bin_duration, 1), height=np.zeros(bin_duration)+np.max(vel_overall_mean+vel_overall_stderr), width=1, color='#009FE3', alpha=.3)
plt.ylabel('Velocity(m/s)')
plt.xlabel('Time[s]')
plt.legend()
plt.title('Velocity Change Around Laser Epoch')
plt.savefig(output_folder + '/velocity'  + '.png', dpi=300)
plt.close()

plt.figure()
plt.plot(SSD_overall_mean,color='#CC6677',alpha=.7)
plt.fill_between(np.arange(minute),y1 = SSD_overall_mean-SSD_overall_stderr,y2=SSD_overall_mean+SSD_overall_stderr,color='#CC6677',alpha=.3)
plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
plt.bar(x=np.arange(bin_duration, 2 * bin_duration, 1), height=np.zeros(bin_duration)+np.max(SSD_overall_mean+SSD_overall_stderr), width=1, color='#009FE3', alpha=.3)
plt.ylabel('SSD(a.u)')
plt.xlabel('Time[s]')
# plt.legend()
plt.title('SSD Change Around Laser Epoch')
plt.savefig(output_folder + '/SSD.png', dpi=300)
plt.close()

#%% Comparison of vigor on/off laser for all behaviors
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Movement_parameters/'
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
clustered_on = {VDB:[],nVDB:[]}
clustered_off ={VDB:[],nVDB:[]}
m_idx = 0
for mouse in stim_day:
    if mouse == 'c488Bm7':continue
    print(mouse)
    predictions = stim_day[mouse]['merged']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    laser_times = []
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        laser_times.extend(np.arange(laser_onset,laser_onset+bin_duration,1))
    for behavior in range(num_of_behaviors):
        behavior_indices = np.nonzero(predictions==behavior)[0]
        for idx in behavior_indices:
            if idx in laser_times:
                laser_on[behavior].append(SSD[idx])
            else:
                laser_off[behavior].append(SSD[idx])

    clustered_predictions = np.copy(predictions)
    clustered_predictions = np.where(clustered_predictions<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        behavior_indices = np.nonzero(clustered_predictions==category)[0]
        for idx in behavior_indices:
            if idx in laser_times:
                clustered_on[category].append(SSD[idx])
            else:
                clustered_off[category].append(SSD[idx])

    m_idx+=1
for behavior in range(num_of_behaviors):
    plt.figure()
    if len(laser_on[behavior]):
        sns.histplot(laser_on[behavior],color = 'blue', alpha=.3, label = 'Laser on',stat='probability',bins = 100)
    if len(laser_off[behavior]) > 0:
        sns.histplot(laser_off[behavior], color='gray', alpha=.3 , label = 'Laser off',stat='probability',bins = 100)
    plt.xlabel('SSD(a.u)')
    plt.title('Distribution of SSD in laser on/off times\n Behavior: '+behaviors[behavior])
    plt.savefig(output_folder+'laser_on_off_'+behaviors[behavior]+'.png', dpi=300)
    plt.close()

for category in [VDB, nVDB]:
    plt.figure()
    sns.histplot(clustered_on[category],color = 'blue', alpha=.3, label = 'Laser on',stat='probability',bins = 100)
    sns.histplot(clustered_off[category], color='gray', alpha=.3 , label = 'Laser off',stat='probability',bins = 100)
    plt.xlabel('SSD(a.u)')
    if category==VDB:
        plt.title('Distribution of SSD index in laser on/off times\n Behavior: Grooming, body licking, wall licking, floor licking')
        plt.savefig(output_folder+'laser_on_off_VDB.png', dpi=300)
    else:
        plt.title('Distribution of SSD index in laser on/off times\n Behavior: Rearing, other, BTC, jump')
        plt.savefig(output_folder + 'laser_on_off_nVDB.png', dpi=300)
    plt.close()


#%%
num_of_stims=9
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Summary/'
counts_on = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
probs_on = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
counts_pre = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
probs_pre = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
counts_post = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
probs_post = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)

shuf_counts_on = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
shuf_probs_on = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
shuf_counts_pre = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
shuf_probs_pre = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
shuf_counts_post = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
shuf_probs_post = np.zeros((num_of_mice,num_of_stims,num_of_behaviors),dtype=np.float32)
distance_pre = np.zeros((num_of_mice,num_of_stims))
distance_post = np.zeros((num_of_mice,num_of_stims))
shuf_distance_pre = np.zeros((num_of_mice,num_of_stims))
shuf_distance_post = np.zeros((num_of_mice,num_of_stims))
plt.figure()
m_idx = 0
mice_list = []
for mouse in stim_day:
    print(mouse)
    mice_list.append(mouse)
    predictions = stim_day[mouse]['merged']
    shuffled_pred = np.random.permutation(predictions)
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for idx in range(laser_onset,laser_onset+bin_duration):
            counts_on[m_idx,  stim_idx,predictions[idx]]+= 1
            shuf_counts_on[m_idx,  stim_idx,shuffled_pred[idx]]+= 1
        probs_on[m_idx, stim_idx] = counts_on[m_idx, stim_idx] / bin_duration
        shuf_probs_on[m_idx, stim_idx] = shuf_counts_on[m_idx, stim_idx] / bin_duration
        for idx in range(laser_onset-bin_duration, laser_onset):
            counts_pre[m_idx, stim_idx, predictions[idx]] += 1
            shuf_counts_pre[m_idx, stim_idx, shuffled_pred[idx]] += 1
        for idx in range(laser_onset+bin_duration, laser_onset+2*bin_duration):
            counts_post[m_idx, stim_idx, predictions[idx]] += 1
            shuf_counts_post[m_idx, stim_idx, shuffled_pred[idx]] += 1

        probs_pre[m_idx, stim_idx] = counts_pre[m_idx, stim_idx] / bin_duration
        probs_post[m_idx, stim_idx] = counts_post[m_idx, stim_idx] / bin_duration
        shuf_probs_pre[m_idx, stim_idx] = shuf_counts_pre[m_idx, stim_idx] / bin_duration
        shuf_probs_post[m_idx, stim_idx] = shuf_counts_post[m_idx, stim_idx] / bin_duration

        distance_pre[m_idx,stim_idx] = stats.wasserstein_distance(np.arange(num_of_behaviors),np.arange(num_of_behaviors),probs_pre[m_idx, stim_idx],probs_on[m_idx, stim_idx])
        shuf_distance_pre[m_idx, stim_idx] = stats.wasserstein_distance(np.arange(num_of_behaviors), np.arange(num_of_behaviors), shuf_probs_pre[m_idx, stim_idx],
                                                               shuf_probs_on[m_idx, stim_idx])

        distance_post[m_idx,stim_idx] = stats.wasserstein_distance(np.arange(num_of_behaviors),np.arange(num_of_behaviors),probs_post[m_idx, stim_idx],probs_on[m_idx, stim_idx])
        shuf_distance_post[m_idx, stim_idx] = stats.wasserstein_distance(np.arange(num_of_behaviors), np.arange(num_of_behaviors), shuf_probs_post[m_idx, stim_idx],
                                                               shuf_probs_on[m_idx, stim_idx])
    m_idx+=1

mean_pre = np.mean(distance_pre,axis = 0)
shuf_mean_pre = np.mean(shuf_distance_pre,axis = 0)
stderr_pre = np.sqrt(np.var(distance_pre,axis=0)/num_of_mice)
shuf_stderr_pre = np.sqrt(np.var(shuf_distance_pre,axis=0)/num_of_mice)

mean_post = np.mean(distance_post,axis = 0)
shuf_mean_post = np.mean(shuf_distance_post,axis = 0)
stderr_post = np.sqrt(np.var(distance_post,axis=0)/num_of_mice)
shuf_stderr_post = np.sqrt(np.var(shuf_distance_post,axis=0)/num_of_mice)

plt.plot(mean_pre,label = '|$\mathcal{D}$$^{REAL}$$_{laser-on}$ - $\mathcal{D}$$^{REAL}$$_{pre-laser}$ |')
plt.fill_between(x = np.arange(num_of_stims),y1 = mean_pre-stderr_pre , y2 = mean_pre+stderr_pre,alpha = .3)
plt.plot(shuf_mean_pre , label = '|$\mathcal{D}$$^{SHUF}$$_{laser-on}$ - $\mathcal{D}$$^{SHUF}$$_{pre-laser}$ |')
plt.fill_between(x = np.arange(num_of_stims),y1 = shuf_mean_pre-shuf_stderr_pre , y2 = shuf_mean_pre+shuf_stderr_pre,alpha = .3)

plt.plot(mean_post,label = '|$\mathcal{D}$$^{REAL}$$_{laser-on}$ - $\mathcal{D}$$^{REAL}$$_{post-laser}$ |')
plt.fill_between(x = np.arange(num_of_stims),y1 = mean_post-stderr_post , y2 = mean_post+stderr_post,alpha = .3)
plt.plot(shuf_mean_post , label = '|$\mathcal{D}$$^{SHUF}$$_{laser-on}$ - $\mathcal{D}$$^{SHUF}$$_{post-laser}$ |')
plt.fill_between(x = np.arange(num_of_stims),y1 = shuf_mean_post-shuf_stderr_post , y2 = shuf_mean_post+shuf_stderr_post,alpha = .3)

plt.legend()
plt.xticks(np.arange(9),np.arange(1,10,1))
plt.grid()
plt.xlabel('# of stimulation')
plt.ylabel('Wasserstein Distance')
plt.title('Changes in distribution over behaviors around laser onset')
plt.savefig(output_folder + 'distribution_distances.png', dpi=300)
plt.close()


