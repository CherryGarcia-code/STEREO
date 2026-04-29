import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import bz2
import seaborn as sns
import matplotlib.pyplot as plt
import helper_functions as hf
import pickle
import scipy.signal as signal
dict_version = 'Dec24'
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CTM_'+dict_version+'.pkl', 'rb')
stim_day = pickle.load(ifile)['drd1_opto']['AloneStim']
ifile.close()
temp_file = open(folder+'opto_alignment_dict.pkl', 'rb')
opto_dict = pickle.load(temp_file)
temp_file.close()
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
FPS=15
second = 15
minute = 60*second
ISI = 1 * minute + 20*second
PRE = 0
DURING = 1
POST = 2
bin_duration = 20 * second
num_of_bins = 3
num_of_behaviors = len(behaviors)
num_of_stims = 9
num_of_mice = len(stim_day.keys())
PATHO_LICKING = 0
NATURAL_LICKING = 1
NO_LICKING = 2
grouping_lut =  np.array([NO_LICKING,NO_LICKING,PATHO_LICKING,PATHO_LICKING,NATURAL_LICKING,NATURAL_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
mm=1/25.4
#%% Fig 6E
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Ethograms/'
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity'] * FPS*10
    SSD = hf.smoothing(np.max(np.vstack([stim_day[mouse]['SSD']['cam1'],stim_day[mouse]['SSD']['cam2']]),axis=0),3)*1000
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    stims = np.empty_like(predictions)
    stims[:]=10
    for stim in stim_times:
        stims[stim:stim+20*second]=9
    predictions = np.reshape(predictions, (1, predictions.shape[0]))
    stims = np.reshape(stims, (1, stims.shape[0]))
    fig,ax = plt.subplots(nrows=4,sharex='all',figsize=(4, 1.6),gridspec_kw={'height_ratios': [1, 4,3,3]})
    ax[1] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,cbar = False,vmin=0, vmax=num_of_behaviors-1, ax=ax[1])
    ax[1].tick_params(left=False, bottom=False)
    plt.sca(ax[1])
    ax[0] = sns.heatmap(stims, yticklabels=[''],cbar = False, cmap=['#B2E2F6','white'], vmin=9, vmax=10,ax= ax[0])
    ax[0].tick_params(left=False, bottom=False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    plt.sca(ax[0])

    ax[2].plot(velocity,lw=.5)
    ax[2].bar(x=np.nonzero(stims == 9)[1], height=np.max(velocity), width=1, color='#B2E2F6' )
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['top'].set_visible(False)
    plt.sca(ax[2])
    plt.yticks([0,1, 2], [0, 1,2],fontsize=8)
    ax[3].plot(SSD,lw=.3,color='gray')
    ax[3].bar(x=np.nonzero(stims==9)[1],height = np.nanmax(SSD),width = 1,color='#B2E2F6')
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['top'].set_visible(False)
    plt.sca(ax[3])
    plt.ylim([0,5])
    plt.yticks([0,5],[0,5],fontsize=8)
    plt.xticks([0,2700],[0,3],rotation=0, fontsize=8)
    fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder + '/' + mouse + '.png', dpi=300)
    plt.savefig(output_folder+'/'+mouse+'.pdf',dpi=300)
    plt.close()
#%% Fig 6F
output_folder = 'output/Optogenetics/Drd1/Summary/'
smooth_window_len = 60
smooth_window = np.ones(smooth_window_len) / smooth_window_len
m_idx=0
for mouse in stim_day:
    print(mouse)
    m_predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
    offset = opto_dict[mouse][1]
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
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

fig,ax = plt.subplots(nrows=2,sharex='all',gridspec_kw={'height_ratios': [1, 20]},figsize=(120*mm,50*mm))
time = np.arange(predictions.shape[1])
for row in range(dist.shape[0]):
    dist[row,:]= signal.filtfilt(smooth_window, 1, dist[row,:])
plt.sca(ax[1])
plt.stackplot(time,dist[0],dist[1],dist[2],dist[3],dist[4],dist[5],dist[6],dist[7],dist[8],labels = behaviors,colors=colors)
plt.xticks(np.arange(0,predictions.shape[1],4500),np.arange(0,16,5),fontsize=8)
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
#%% Fig 6G
dominant_per_stim = np.zeros((num_of_mice,num_of_stims))
dist_per_m = np.zeros((num_of_mice,num_of_behaviors))
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Summary/'
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    dom_behaviors = []
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        behaviors_per_stim, count_per_stim=np.unique(predictions[laser_onset:laser_onset+bin_duration],return_counts=True)
        dom_behavior = behaviors_per_stim[np.argmax(count_per_stim)]
        dominant_per_stim[m_idx,stim_idx]=dom_behavior
        dom_behaviors.append(dom_behavior)
    print(mouse,dom_behaviors)
    m_idx += 1
for stim_idx in range(num_of_stims):
    behaviors_per_m,count_per_m = np.unique(dominant_per_stim[:,stim_idx],return_counts=True)
    dist_per_m[stim_idx,behaviors_per_m.astype(np.int8)] = count_per_m

dominant_per_stim=grouping_lut[dominant_per_stim.astype(np.int8)]
population_distribution = []
chance_level = 1.0/3
MIXED=3
for m_idx in range(num_of_mice):
    patho_licking = np.count_nonzero(dominant_per_stim[m_idx,:] == PATHO_LICKING)/num_of_stims
    natural_licking = np.count_nonzero(dominant_per_stim[m_idx, :] == NATURAL_LICKING) / num_of_stims
    no_licking = np.count_nonzero(dominant_per_stim[m_idx, :] == NO_LICKING) / num_of_stims
    m_tendency = np.array([patho_licking,natural_licking,no_licking])
    print(m_tendency,chance_level)
    m_tendency = m_tendency>chance_level
    print(m_tendency)
    if np.count_nonzero(m_tendency)==2:
        population_distribution.append(MIXED)
    else:
        if m_tendency[PATHO_LICKING]:
            population_distribution.append(PATHO_LICKING)
        elif m_tendency[NATURAL_LICKING]:
            population_distribution.append(NATURAL_LICKING)
        elif m_tendency[NO_LICKING]:
            population_distribution.append(NO_LICKING)

print(population_distribution)

population_distribution = np.unique(population_distribution,return_counts=True)[1]
plt.figure(figsize=(2.4,1.6),frameon=False)
plt.pie(population_distribution, colors =  ["#d73027", "#c4a7e7", "#1f4e79","#d3d3d3"],autopct='%.0f%%')
plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
plt.savefig(output_folder+'Population_distribution.png',dpi=300)
plt.savefig(output_folder+'Population_distribution.pdf',dpi=300)
#%% Fig 6H,J
stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,num_of_behaviors),dtype=np.float32)
clustered_stim_epoch_behavior = np.zeros((num_of_bins, num_of_stims , num_of_mice,3),dtype=np.float32)
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
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
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Summary/'
for behavior in range(num_of_behaviors):
    plt.figure(figsize=(2,1.2),frameon=False)
    plt.plot(np.mean(stim_epoch_behavior[:,:,:,behavior],axis=1),alpha=0.3,color='gray',marker='o',markersize=2,lw=1)
    plt.bar(1,height=1 , color='#009FE3', alpha=.3)
    plt.errorbar(np.arange(3),y=np.mean(stim_epoch_behavior[:,:,:,behavior],axis=(1,2)),
                 yerr=np.sqrt(np.var(stim_epoch_behavior[:,:,:,behavior])/(stim_epoch_behavior.shape[1]*stim_epoch_behavior.shape[2])).T,
                 color=colors[behavior],capsize=2,capthick=1,lw=1)
    plt.xticks(np.arange(3),['Prior','During','Post'],fontsize=8)
    plt.ylabel('% Time spent',fontsize=10)
    if behavior==4:
        plt.yticks([0,0.5,1],[0,50,100],fontsize=8)
        plt.ylim([0, 1])
    elif behavior==2:
        plt.yticks([0, 0.25, 0.5], [0,25,50], fontsize=8)
        plt.ylim([0, .5])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder + '/V_graph_'+behaviors[behavior]+ '.png', dpi=300)
    plt.savefig(output_folder + '/V_graph_' + behaviors[behavior] + '.pdf', dpi=300)
    plt.close()
#%% Fig 6I,K
dynamics = np.zeros((num_of_mice,num_of_stims,num_of_behaviors,minute),dtype=np.float32)
clustered_dynamics = np.zeros((num_of_mice,num_of_stims,3,minute),dtype=np.float32)
m_idx = 0
for mouse in stim_day:
    print(mouse)
    predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
    for stim_idx in range(stim_times.size):
        laser_onset = stim_times[stim_idx]
        for behavior in range(num_of_behaviors):
            dynamics[m_idx , stim_idx, behavior,:] = predictions[laser_onset-bin_duration:laser_onset+(2*bin_duration)]==behavior
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Dynamics/'
for behavior in range(num_of_behaviors):
    psth = np.mean(dynamics[:,:,behavior,:],axis=1)
    for m in range(num_of_mice):
        psth[m] = hf.smoothing(psth[m],5)
    mean_psth = np.mean(psth,axis=0)
    stderr = np.sqrt(np.var(psth,axis=0)/psth.shape[0])
    plt.figure(figsize=(2,1.35),frameon=False)
    plt.bar(x=bin_duration, height=np.ones(bin_duration), width=bin_duration, align='edge', color='#B2E2F7')
    plt.plot(mean_psth,color=colors[behavior],lw=1)
    plt.fill_between(np.arange(minute),y1 = mean_psth-stderr,y2=mean_psth+stderr,color=colors[behavior],alpha=.3)
    # plt.xticks(np.arange(0, minute, 5 * second), (np.arange(-bin_duration, 2 * bin_duration, 5 * second) / second).astype(int))

    plt.ylabel('Probability',fontsize=10)
    plt.xlabel('Time[s]',fontsize=8)
    plt.xticks([0,150,300,450,600,750,900], [-20,-10,0,10,20,30,40], fontsize=8)
    if behavior == 4:
        plt.yticks([0, 0.5, 1], [0, 0.5, 1], fontsize=8)
        plt.ylim([0, 1])
    elif behavior == 2:
        plt.yticks([0, 0.25, 0.5], [0, 0.25, 0.5], fontsize=8)
        plt.ylim([0, .5])

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plt.title('Dynamics of Change in behavior around laser epoch\n Behaviors: '+behaviors[behavior])
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0)
    plt.savefig(output_folder + '/dynamics_'+behaviors[behavior]  + '.png', dpi=300)
    plt.savefig(output_folder + '/dynamics_' + behaviors[behavior] + '.pdf', dpi=300)
    plt.close()
#%% Fig 6L,M
output_folder = root_folder+'/Figures/Optogenetics/Drd1/Movement_parameters/'
laser_off ={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
laser_on = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
m_idx = 0
for mouse in stim_day:
    if mouse == 'c488Bm7':continue
    print(mouse)
    predictions = stim_day[mouse]['merged']['predictions']['smartMerge']
    offset = opto_dict[mouse][1]
    velocity = stim_day[mouse]['topcam']['velocity']*FPS*10
    SSD = np.max(np.vstack([stim_day[mouse]['SSD']['cam1'], stim_day[mouse]['SSD']['cam2']]), axis=0)*1000
    first_stim = 3 * minute - int((offset / 1000.0) * 15)
    stim_times = np.arange(first_stim, 14 * minute, ISI)
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
    plt.figure(figsize=(2.5,2), frameon=False)
    sns.kdeplot(x=laser_off[behavior][:, 0],y=laser_off[behavior][:,1],label ='Laser off',color='gray',alpha=.5,fill=True)
    sns.kdeplot(x=laser_on[behavior][:, 0], y=laser_on[behavior][:, 1], label='Laser on',color=colors[behavior],alpha=.5,fill=True)
    plt.ylabel('Velocity(cm/s)',fontsize=10)
    plt.xlabel('SSD (a.u.)',fontsize=10)
    handles, labels = plt.gca().get_legend_handles_labels()
    colors_tag = ['gray', colors[behavior]]
    plt.legend(handles, labels, labelcolor=colors_tag, fontsize=8,frameon=False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.xticks(fontsize=8)
    if behavior ==4:
        plt.ylim([0,.7])
        plt.yticks([0,0.3,0.6],[0,0.3,0.6],fontsize=8)
    if behavior == 2:
        plt.yticks([0, 0.75, 1.5], [0, 0.75, 1.5], fontsize=8)
    lims_idx+=1
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
    plt.savefig(output_folder+'SSD_laser_on_off_'+behaviors[behavior]+'_corr.png', dpi=300)
    plt.savefig(output_folder + 'SSD_laser_on_off_' + behaviors[behavior] + '_corr.pdf', dpi=300)
    plt.close()