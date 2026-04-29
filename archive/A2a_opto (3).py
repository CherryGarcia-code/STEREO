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
ifile = bz2.BZ2File(folder + 'CTM_Aug24.pkl', 'rb')
stim_days = pickle.load(ifile)['a2a_opto']
stim_days = {key:stim_days[key] for key in ['cocaine6laserStim','cocaine8laserStim']}
ifile.close()
print('***All dictionaries were loaded***')
# behaviors = ['Jump','Back to camera','Floor licking', 'Wall licking','Grooming', 'Body licking','Rearing','Locomotion','Stationary']
# colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163','#969595', '#b4a7d6']
# colors = ['#D3D3D3', '#696969', '#CD5C5C', '#FF7F7F', '#DA70D6', '#BA55D3', '#9370DB','#87CEFA', '#ADD8E6']

# color-blind friendly 
behaviors = ['Jump', 'Back to camera', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
# colors = ['#D3D3D3', '#696969',         '#ea5f94',       '#fa8775',      '#ffd700',    '#ffb14e',   '#9d02d7',   '#0000ff',   '#cd34b5']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
# Specifically:
# Floor licking: '#D32F2F' (strong red)
# Wall licking: '#FFCDD2' (weaker red)
# Grooming: '#9467BD' (purple)
# Body licking: '#8C564B' (brownish-purple)
# Rearing: '#009E73' (green)
SSD_color = '#ACE1AF'
velocity_color = '#FFC87C'#'#FFC127'
FPS=15
sample_rate=15
second = 15
minute = 60*second
ISI = 1 * minute + 20*second
mpl.rcParams["mathtext.fontset"] = 'cm'
mpl.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Arial'
# Adjust text sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
LARGE_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # Controls default text size
plt.rc('axes', titlesize=MEDIUM_SIZE)    # Controls title size
plt.rc('axes', labelsize=MEDIUM_SIZE)    # Controls x and y label size
plt.rc('xtick', labelsize=SMALL_SIZE)    # Controls x tick label size
plt.rc('ytick', labelsize=SMALL_SIZE)    # Controls y tick label size
plt.rc('legend', fontsize=SMALL_SIZE)    # Controls legend font size
PRE = 0
DURING = 1
POST = 2
VDB = 0
nVDB = 1
bin_duration = 20 * second
num_of_bins = 3
num_of_behaviors = len(behaviors)
num_of_stims = 14
num_of_mice = len(stim_days['cocaine6laserStim'].keys())
RNN_offset = 7


#%% Plot ethograms with stimulation ticks

SSD_color = '#1f4e79'
velocity_color = '#3399cc'  # '#FFC127'


output_folder = root_folder + '/Figures/Optogenetics/A2a/Ethograms/'
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)


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
        
        # Set figure size in inches (400 points / 72, 120 points / 72)
        fig, ax = plt.subplots(nrows=4, sharex='all', figsize=(3.9, 1.6), gridspec_kw={'height_ratios': [1, 4, 3, 3]})
        
        # cbar_ax = fig.add_axes([.91, .3, .01, .4])
        # ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,
        #                     cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5, 'drawedges': True},
        #                     cbar_ax=cbar_ax, vmin=0, vmax=num_of_behaviors - 1, ax=ax[1])
        # cbar = ax[1].collections[0].colorbar
        # cbar.set_ticklabels(behaviors)
        
        # plt.sca(ax[1])
        # # plt.ylabel('Behavior', rotation=0)
        
        # ax[0] = sns.heatmap(stims, yticklabels=[''], cbar=False, cmap=['#B2E2F6', 'white'], vmin=num_of_behaviors + 1, vmax=num_of_behaviors + 2, ax=ax[0])
        # ax[0].tick_params(left=False, bottom=False)
        # ax[0].spines['left'].set_visible(True)
        # ax[0].spines['bottom'].set_visible(False)
        # ax[0].spines['top'].set_visible(True)
        # ax[0].spines['right'].set_visible(True)
        # Removed cbar_ax and cbar_kws
        ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,
                            cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[1])

        plt.sca(ax[1])
        # plt.ylabel('Behavior', rotation=0)

        ax[0] = sns.heatmap(stims, yticklabels=[''], cbar=False, cmap=['#B2E2F6', 'white'], vmin=num_of_behaviors + 1, vmax=num_of_behaviors + 2, ax=ax[0])
        ax[0].tick_params(left=False, bottom=False)
        ax[0].spines['left'].set_visible(True)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['top'].set_visible(True)
        ax[0].spines['right'].set_visible(True)

        
        plt.sca(ax[0])
        # plt.ylabel('Stim', rotation=0)
        
        ax[2].plot(velocity * FPS, c=velocity_color, linewidth=0.3)  # Adjusted line width
        ax[2].bar(x=np.nonzero(stims == 9)[1], height=np.max(velocity) * FPS, width=1, color='#009FE3', alpha=.3)
        ax[2].spines['right'].set_visible(False)
        ax[2].spines['top'].set_visible(False)
        
        plt.sca(ax[2])
        # plt.ylabel('Velocity (m/s)', rotation=0)
        
        ax[3].plot(SSD, c=SSD_color, alpha=.7, linewidth=0.3)  # Adjusted line width
        ax[3].bar(x=np.nonzero(stims == 9)[1], height=np.max(SSD[~np.isnan(SSD)]) - np.min(SSD[~np.isnan(SSD)]), bottom=np.min(SSD[~np.isnan(SSD)]), width=1, color='#009FE3', alpha=.3)
        ax[3].spines['right'].set_visible(False)
        ax[3].spines['top'].set_visible(False)
        
        plt.sca(ax[3])
        # plt.ylabel('SSD (a.u.)', rotation=0)
        
        interval = 10 * 60 * FPS  # 10 minutes interval in frames
        plt.xticks(np.arange(0, predictions.shape[1], interval), (np.arange(0, predictions.shape[1], interval) / (60 * FPS)).astype(int), rotation=0, fontsize=8)
        plt.xlabel('Time (minutes)')
        
        fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
        plt.savefig(output_folder + '/' + mouse + '_' + t + '.pdf', dpi=300)
        plt.close()


#%% V graphs-VDB/nVDB
#%% Dynamics - Pathological/natural/no licking
output_folder = root_folder+'/Figures/Optogenetics/A2a/Dynamics/'
dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
clustered_dynamics = np.zeros((num_of_mice,num_of_stims,3,minute),dtype=np.float32)
m_idx = 0
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
outliers = []

for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(stim_days['cocaine8laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']
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
num_of_mice_wo_outliers = num_of_mice-len(outliers)
dynamics = dynamics[:num_of_mice_wo_outliers,:,:,:]
clustered_dynamics = clustered_dynamics[:num_of_mice_wo_outliers,:,:,:]
titles = ['Pathological licking','Natural licking','No licking']
grouped_colors = ['red','purple','blue']
for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
    psth = np.mean(clustered_dynamics[:,:,category,:],axis=1)
    for m in range(num_of_mice_wo_outliers):
        psth[m] = hf.smoothing(psth[m],5)
    mean_psth = np.mean(psth,axis=0)
    stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
    plt.figure()
    plt.plot(mean_psth,color=grouped_colors[category],alpha=.7)
    plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color=grouped_colors[category],alpha=.3)
    plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
    plt.bar(x=np.arange(bin_duration,2*bin_duration,1),height=np.ones(bin_duration) ,width=1, color='#009FE3', alpha=.3)
    plt.ylabel('Fraction of mice')
    plt.xlabel('Time[s]')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    plt.title(titles[category])
    plt.savefig(output_folder + '/dynamics_'+titles[category].replace(' ','_')+'.png', dpi=300)
    plt.close()
#%% V graphs - all behaviors
output_folder = root_folder+'/Figures/Optogenetics/A2a/Summary/'
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


output_folder = root_folder+'/Figures/Optogenetics/A2a/Dynamics/'
dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
clustered_dynamics = np.zeros((num_of_mice,num_of_stims,2,minute),dtype=np.float32)
m_idx = 0
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(stim_days['cocaine8laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']
    velocity = stim_days[t][mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            dynamics[m_idx , stim_idx, behavior,:] = predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==behavior
        clustered_predictions = np.copy(predictions)
        clustered_predictions = np.where((clustered_predictions<=3) & (clustered_predictions>1),VDB,nVDB)
        for category in [VDB,nVDB]:
            clustered_dynamics[m_idx , stim_idx, category,:] = clustered_predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==category
    m_idx+=1
num_of_mice_wo_outliers = num_of_mice-len(outliers)
dynamics = dynamics[:num_of_mice_wo_outliers,:,:,:]
clustered_dynamics = clustered_dynamics[:num_of_mice_wo_outliers,:,:,:]
for category in [VDB,nVDB]:
    psth = np.mean(clustered_dynamics[:,:,category,:],axis=1)
    for m in range(num_of_mice_wo_outliers):
        psth[m] = hf.smoothing(psth[m],5)
    mean_psth = np.mean(psth,axis=0)
    stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
    plt.figure(figsize=(1.6,1.45))
    plt.plot(mean_psth,color='#D73027')
    plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color='#D73027',alpha=.3)
    plt.xticks(np.arange(0, minute, 10 * second), (np.arange(-bin_duration, 2 * bin_duration, 10 * second) / second).astype(int))
    plt.bar(x=np.arange(bin_duration,2*bin_duration,1),height=np.ones(bin_duration) ,width=1, color='#009FE3', alpha=.3)
    plt.ylabel('Fraction of mice')

    plt.xlabel('Time[s]')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    sns.despine()
    if category==VDB:
        # plt.title('Dynamics of Change in behavior around laser epoch\n Behaviors: wall licking,floor licking')
        plt.savefig(output_folder + '/dynamics_VDB.pdf', dpi=300)
    else:
        plt.title('Dynamics of Change in behavior around laser epoch\n Behaviors: Rearing, Other,BTC,Jump')
        plt.savefig(output_folder + '/dynamics_nVDB.pdf', dpi=300)
    plt.close()
#%% Dynamics- all behaviors
output_folder = root_folder+'/Figures/Optogenetics/A2a/Dynamics/'
for behavior in range(num_of_behaviors):
    psth = np.mean(dynamics[:,:,behavior,:],axis=1)
    for m in range(num_of_mice_wo_outliers):
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
output_folder = root_folder+'/Figures/Optogenetics/A2a/Movement_parameters/'
num_of_cams=2
velocity_matrix = np.zeros((num_of_mice,num_of_stims,minute),dtype=np.float32)
SSD_matrix = np.zeros((num_of_mice,num_of_stims,minute),dtype=np.float32)
m_idx = 0
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(stim_days['cocaine8laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']
    velocity = stim_days[t][mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        velocity_matrix[m_idx,stim_idx,:]=velocity[laser_onset-bin_duration:laser_onset+(2*bin_duration)]
        SSD_matrix[m_idx,stim_idx,:]=SSD[laser_onset-bin_duration:laser_onset+(2*bin_duration)]

    m_idx+=1
num_of_mice_wo_outliers = num_of_mice-len(outliers)
velocity_matrix = velocity_matrix[:num_of_mice_wo_outliers,:,:]
SSD_matrix = SSD_matrix[:num_of_mice_wo_outliers,:,:]

velocity_matrix *=FPS
vel_mouse_mean = np.mean(velocity_matrix[:,:,:],axis=1)
SSD_mouse_mean = np.nanmean(SSD_matrix[:,:,:],axis=1)
vel_overall_mean = np.mean(vel_mouse_mean,axis=0)
vel_overall_stderr = np.sqrt(np.var(vel_mouse_mean,axis=0)/vel_mouse_mean.shape[0])
SSD_overall_mean = np.nanmean(SSD_mouse_mean,axis=0)
SSD_overall_stderr = np.sqrt(np.nanvar(SSD_mouse_mean,axis=0)/SSD_mouse_mean.shape[0])


plt.figure()
plt.plot(vel_overall_mean,color=velocity_color)
plt.fill_between(np.arange(minute),y1 = vel_overall_mean-vel_overall_stderr,y2=vel_overall_mean+vel_overall_stderr,color=velocity_color,alpha=.3)
plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
plt.bar(x=np.arange(bin_duration, 2 * bin_duration, 1), height=np.zeros(bin_duration)+np.max(vel_overall_mean+vel_overall_stderr), width=1, color='#009FE3', alpha=.3)
plt.ylabel('Velocity(m/s)')
plt.xlabel('Time[s]')
plt.legend()
plt.title('Velocity Change Around Laser Epoch')
plt.savefig(output_folder + '/velocity'  + '.png', dpi=300)
plt.close()

plt.figure()
plt.plot(SSD_overall_mean,color=SSD_color,alpha=.7)
plt.fill_between(np.arange(minute),y1 =SSD_overall_mean-SSD_overall_stderr,y2=SSD_overall_mean+SSD_overall_stderr,color=SSD_color,alpha=.3)
plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))
plt.bar(x=np.arange(bin_duration, 2 * bin_duration, 1), height=np.zeros(bin_duration)+np.max(SSD_overall_mean+SSD_overall_stderr)-np.min(SSD_overall_mean-SSD_overall_stderr), bottom=np.zeros(bin_duration)+np.min(SSD_overall_mean-SSD_overall_stderr), width=1, color='#009FE3', alpha=.3)
plt.ylabel('Vigor(a.u)')
plt.xlabel('Time[s]')
plt.legend()
plt.title('Vigor Change Around Laser Epoch')
plt.savefig(output_folder + '/SSD' + '.png', dpi=300)
plt.close()

#%% Comparison of vigor on/off laser for all behaviors
output_folder = root_folder+'/Figures/Optogenetics/A2a/Movement_parameters/'
first_stim = 10 * minute - RNN_offset
stim_times = np.arange(first_stim, 28 * minute, ISI)
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
clustered_on = {VDB:[],nVDB:[]}
clustered_off ={VDB:[],nVDB:[]}
m_idx = 0
outliers = []
for mouse in stim_days['cocaine6laserStim']:
    print(mouse)
    if np.count_nonzero(stim_days['cocaine6laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t='cocaine6laserStim'
    elif np.count_nonzero(stim_days['cocaine8laserStim'][mouse]['merged'][:first_stim]<=3)/first_stim>=0.5:
        t = 'cocaine8laserStim'
    else:
        m_idx += 1
        outliers.append(mouse)
        continue
    predictions = stim_days[t][mouse]['merged']
    velocity = stim_days[t][mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_days[t][mouse]['SSD']['cam1'], stim_days[t][mouse]['SSD']['cam2']]), axis=0)
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
    plt.legend()
    plt.savefig(output_folder+'laser_on_off_'+behaviors[behavior]+'.png', dpi=300)
    plt.close()

for category in [VDB, nVDB]:
    plt.figure()
    sns.histplot(clustered_on[category],color = 'blue', alpha=.3, label = 'Laser on',stat='probability',bins = 100)
    sns.histplot(clustered_off[category], color='gray', alpha=.3 , label = 'Laser off',stat='probability',bins = 100)
    plt.xlabel('SSD(a.u)')
    plt.legend()
    if category==VDB:
        plt.suptitle('Distribution of vigor index in laser on/off times\n Behavior: Grooming, body licking, wall licking, floor licking')
        plt.savefig(output_folder+'laser_on_off_VDB.png', dpi=300)
    else:
        plt.suptitle('Distribution of vigor index in laser on/off times\n Behavior: Rearing, other, BTC, jump')
        plt.savefig(output_folder + 'laser_on_off_nVDB.png', dpi=300)
    plt.close()

#%%
output_folder = 'Figures/'+model+'/Optogenetics/A2a/Summary/'
for t in ['cocaine6laserStim' ,'cocaine8laserStim' ]:
    plt.figure()
    m_idx=0
    counts_on = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    probs_on = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    counts_pre = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    probs_pre = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    counts_post = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    probs_post = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_counts_on = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_probs_on = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_counts_pre = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_probs_pre = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_counts_post = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    shuf_probs_post = np.zeros((num_of_mice, num_of_stims, num_of_behaviors), dtype=np.float32)
    distance_pre = np.zeros((num_of_mice, num_of_stims))
    distance_post = np.zeros((num_of_mice, num_of_stims))
    shuf_distance_pre = np.zeros((num_of_mice, num_of_stims))
    shuf_distance_post = np.zeros((num_of_mice, num_of_stims))
    mice_list = []
    for mouse in stim_days[t]:
        print(mouse)
        predictions = stim_days[t][mouse]['merged']
        mice_list.append(mouse)
        shuffled_pred = np.random.permutation(predictions)
        first_stim = 10 * minute - RNN_offset
        stim_times = np.arange(first_stim, 28 * minute, ISI)
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
    plt.xticks(np.arange(num_of_stims),np.arange(1,num_of_stims+1,1))
    plt.grid()
    plt.xlabel('# of stimulation')
    plt.ylabel('Wasserstein_Distance')
    plt.title('Changes in distribution over behaviors around laser onset\n'+t)
    plt.savefig(output_folder + 'distribution_distances_'+t+'.png', dpi=300)
    plt.close()


