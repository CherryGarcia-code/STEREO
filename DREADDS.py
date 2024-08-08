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
import matplotlib as mpl
import networkx as nx
model = 'model1'
folder = 'Data/'+model+'/'
ifile = bz2.BZ2File(folder + 'CMT.pkl', 'rb')
# ***CHANGE THE COHORT HERE***
cohort = 'drd1_hm4di'
MT = pickle.load(ifile)[cohort]
ifile.close()
possible_trials= {'cocaineOnly':['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','cocaine6','cocaine7','cocaine8','cocaine9','cocaine10'],
                                'cocaineCNO':['cocaine6afterCNO','cocaine7afterCNO', 'cocaine8afterCNO', 'cocaine9afterCNO']}
sandwich_days = {}
for m in MT.keys():
    trials = list(MT[m].keys())
    for t_idx in range(1, len(trials) - 1):
        if trials[t_idx] in possible_trials['cocaineCNO'] and trials[t_idx + 1] in possible_trials['cocaineOnly'] and \
                (trials[t_idx - 1] in possible_trials['cocaineOnly'] or (
                        'CNOonly' in trials[t_idx - 1] and trials[t_idx - 2] in possible_trials['cocaineOnly'])):
            sandwich_days[m]={}
            sandwich_days[m]['CNO'] = MT[m][trials[t_idx]]
            sandwich_days[m]['post'] = MT[m][trials[t_idx+1]]
            if trials[t_idx - 1] in possible_trials['cocaineOnly']:
                print(m, trials[t_idx-1],trials[t_idx],trials[t_idx+1])
                sandwich_days[m]['pre']  = MT[m][trials[t_idx - 1]]

            else:
                print(m, trials[t_idx - 2], trials[t_idx], trials[t_idx + 1])
                sandwich_days[m]['pre']  = MT[m][trials[t_idx - 2]]
            break
del MT
print('***All dictionaries were loaded***')
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other',  'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#B2B2B2',  '#b4a7d6']
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
num_of_days = 3
num_of_behaviors = len(behaviors)
num_of_mice = len(list(sandwich_days.keys()))
RNN_offset = 7
stationary_theta = 0.021/float(FPS)
locomotive = 0
stationary = 1
# outlier removal
del sandwich_days['c528m10']
del sandwich_days['c548m1']
#%%Representative ethograms
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Ethograms/'
for m in sandwich_days:
    fig , ax = plt.subplots(nrows=3,sharex='all',figsize=(18,6))
    pre = np.copy(sandwich_days[m]['pre']['merged'])
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    post = np.copy(sandwich_days[m]['post']['merged'])
    pre = np.reshape(pre, (1, pre.shape[0]))
    CNO = np.reshape(CNO, (1, CNO.shape[0]))
    post = np.reshape(post, (1, post.shape[0]))
    cbar_ax = fig.add_axes([.88, .3, .01, .4])
    ax[0] = sns.heatmap(pre, yticklabels=[''], cmap=colors,
                        cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5, 'drawedges': True},
                        cbar_ax=cbar_ax, vmin=0, vmax=num_of_behaviors - 1, ax=ax[0])
    plt.sca(ax[0])
    plt.ylabel('Pre-CNO',va='center')
    cbar = ax[0].collections[0].colorbar
    cbar.set_ticklabels(behaviors)
    ax[1] = sns.heatmap(CNO, yticklabels=[''], cmap=colors,cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[1])
    plt.sca(ax[1])
    plt.ylabel('During-CNO', va='center')
    ax[2] = sns.heatmap(post, yticklabels=[''], cmap=colors,cbar = False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[2])
    plt.sca(ax[2])
    plt.ylabel('Post-CNO', va='center')
    T = np.max([pre.size,CNO.size,post.size])
    plt.sca(ax[2])
    plt.xticks(np.arange(0,T,900),(np.arange(RNN_offset,T+RNN_offset,900)/900).astype(int),rotation=0)
    plt.xlabel('Time (m)')
    if '_' in cohort:
        title_cohort = cohort.split('_')[0]
        title_type = cohort.split('_')[1]
        plt.suptitle(title_cohort+'$_{'+title_type+'}$ - '+m)
    else:
        plt.suptitle('control , ' + m)
    fig.tight_layout(rect=[0.02, 0.01, .86, 0.95], pad=1)
    plt.savefig(output_folder+m+'.png',dpi=300)
    plt.close()
#%%V graphs
num_of_mice = len(list(sandwich_days.keys()))
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/wo_stat_loco_separation/'
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days))
clustered_fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_days))
m_idx = 0
for m in sandwich_days:
    pre = np.copy(sandwich_days[m]['pre']['merged'])
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    post = np.copy(sandwich_days[m]['post']['merged'])
    for b in range(num_of_behaviors):
        fraction_of_frames[m_idx , b , PRE] = np.count_nonzero(pre==b)/pre.size
        fraction_of_frames[m_idx, b, DURING] = np.count_nonzero(CNO == b) / CNO.size
        fraction_of_frames[m_idx, b, POST] = np.count_nonzero(post == b) / post.size
    clustered_pre = np.where(pre<=3,VDB,nVDB)
    clustered_CNO = np.where(CNO<=3,VDB,nVDB)
    clustered_post = np.where(post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        clustered_fraction_of_frames[m_idx,category,PRE] = np.count_nonzero(clustered_pre==category)/pre.size
        clustered_fraction_of_frames[m_idx, category, DURING] = np.count_nonzero(clustered_CNO == category) / CNO.size
        clustered_fraction_of_frames[m_idx, category, POST] = np.count_nonzero(clustered_post == category) / post.size
    m_idx+=1
print(fraction_of_frames)
for b in range(num_of_behaviors):
    plt.figure()
    mean = np.mean(fraction_of_frames[:,b,:],axis=0)
    stderr = np.sqrt(np.var(fraction_of_frames[:, b, :], axis=0)/num_of_mice)
    plt.plot(fraction_of_frames[:,b,:].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean.T, color=colors[b])
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylabel('Fraction of frames')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: '+behaviors[b])
    plt.savefig(output_folder + '/V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

plt.figure()
mean = np.mean(clustered_fraction_of_frames[:,VDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,VDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,VDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('Fraction of frames')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming, body licking , floor licking , wall licking')
plt.savefig(output_folder + '/V_graph_VDB.png', dpi=300)
plt.close()

plt.figure()
mean = np.mean(clustered_fraction_of_frames[:,nVDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,nVDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,nVDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('Fraction of frames')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing, BTC, other, jump')
plt.savefig(output_folder + '/V_graph_nVDB.png', dpi=300)
plt.close()
#%%V graphs on bins
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/wo_stat_loco_separation/'
num_of_bins = 3
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days,num_of_bins))
clustered_fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_days,num_of_bins))
m_idx = 0
for m in sandwich_days:
    pre = np.copy(sandwich_days[m]['pre']['merged'])
    edges_pre = np.linspace(0,pre.size,num_of_bins+1,dtype=int)
    print(edges_pre)
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    edges_CNO = np.linspace(0, CNO.size, num_of_bins+1, dtype=int)
    print(edges_CNO)
    post = np.copy(sandwich_days[m]['post']['merged'])
    edges_post = np.linspace(0, post.size, num_of_bins+1, dtype=int)
    print(edges_post)
    for b in range(num_of_behaviors):
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            fraction_of_frames[m_idx , b , PRE,edge] = np.count_nonzero(pre[bin_start:bin_end]==b)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, DURING,edge] = np.count_nonzero(CNO[bin_start:bin_end] == b) / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, POST,edge] = np.count_nonzero(post[bin_start:bin_end] == b) / bin_size
    clustered_pre = np.where(pre<=3,VDB,nVDB)
    clustered_CNO = np.where(CNO<=3,VDB,nVDB)
    clustered_post = np.where(post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx,category,PRE,edge] = np.count_nonzero(clustered_pre[bin_start:bin_end]==category)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx, category, DURING,edge] = np.count_nonzero(clustered_CNO[bin_start:bin_end] == category) / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge+1]
            bin_size = bin_end-bin_start
            clustered_fraction_of_frames[m_idx, category, POST,edge] = np.count_nonzero(clustered_post[bin_start:bin_end] == category) / bin_size
    m_idx+=1

for b in range(num_of_behaviors):
    fig,ax = plt.subplots(ncols=3,sharey='all')
    for bin in range(num_of_bins):
        plt.sca(ax[bin])
        mean = np.mean(fraction_of_frames[:,b,:,bin],axis=0)
        stderr = np.sqrt(np.var(fraction_of_frames[:, b, :,bin], axis=0)/num_of_mice)
        plt.plot(fraction_of_frames[:,b,:,bin].T, alpha=0.3, color='gray', marker='o')
        plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
        plt.plot(mean.T, color=colors[b])
        plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
        plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
        plt.yticks([0,0.5,1],[0,0.5,1])
        plt.ylim([0,1])
        axes = plt.gca()
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.title('Bin #'+str(bin))
    ax[0].set_ylabel('Fraction of frames')
    plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: '+behaviors[b])
    plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
    plt.savefig(output_folder + '/bins_V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

fig,ax = plt.subplots(ncols=3,sharey='all')
for bin in range(num_of_bins):
    plt.sca(ax[bin])
    mean = np.mean(clustered_fraction_of_frames[:,VDB,:,bin],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,VDB,:,bin], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,VDB,:,bin].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean, color='#730AFF')
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])

    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Bin #' + str(bin))
ax[0].set_ylabel('Fraction of frames')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming, body licking , floor licking , wall licking')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.savefig(output_folder + '/bins_V_graph_VDB.png', dpi=300)
plt.close()

fig,ax = plt.subplots(ncols=3,sharey='all')
for bin in range(num_of_bins):
    plt.sca(ax[bin])
    mean = np.mean(clustered_fraction_of_frames[:,nVDB,:,bin],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,nVDB,:,bin], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,nVDB,:,bin].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean, color='#730AFF')
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Bin #' + str(bin))
ax[0].set_ylabel('Fraction of frames')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing, BTC, other, jump')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.savefig(output_folder + '/bins_V_graph_nVDB.png', dpi=300)
plt.close()

#%% Dynamics of CNO effect on stereotypies

output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/wo_stat_loco_separation/'
num_of_bins = 30
cutoff = 15*minute
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days,num_of_bins))
clustered_fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_days,num_of_bins))

m_idx = 0
for m in sandwich_days:
    pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    edges_pre = np.linspace(0,pre.size,num_of_bins+1,dtype=int)
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    edges_CNO = np.linspace(0, CNO.size, num_of_bins+1, dtype=int)
    post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    edges_post = np.linspace(0, post.size, num_of_bins+1, dtype=int)

    for b in range(num_of_behaviors):
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            fraction_of_frames[m_idx , b , PRE,edge] = np.count_nonzero(pre[bin_start:bin_end]==b)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, DURING,edge] = np.count_nonzero(CNO[bin_start:bin_end] == b)/bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            fraction_of_frames[m_idx, b, POST,edge] = np.count_nonzero(post[bin_start:bin_end] == b)/bin_size

    clustered_pre = np.where(pre<=3,VDB,nVDB)
    clustered_CNO = np.where(CNO<=3,VDB,nVDB)
    clustered_post = np.where(post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end - bin_start
            clustered_fraction_of_frames[m_idx,category,PRE,edge] = np.count_nonzero(clustered_pre[bin_start:bin_end]==category)/bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge+1]
            bin_size = bin_end - bin_start
            clustered_fraction_of_frames[m_idx, category, DURING,edge] = np.count_nonzero(clustered_CNO[bin_start:bin_end] == category)/bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge+1]
            bin_size = bin_end - bin_start
            clustered_fraction_of_frames[m_idx, category, POST,edge] = np.count_nonzero(clustered_post[bin_start:bin_end] == category)/bin_size

    m_idx+=1

for b in range(num_of_behaviors):
    plt.figure()
    pre_mean = np.mean(fraction_of_frames[:,b,PRE,:],axis=0)
    pre_stderr = np.sqrt(np.var(fraction_of_frames[:,b,PRE,:],axis=0)/num_of_mice)
    CNO_mean = np.mean(fraction_of_frames[:,b,DURING,:],axis=0)
    CNO_stderr = np.sqrt(np.var(fraction_of_frames[:,b,DURING,:],axis=0)/num_of_mice)
    post_mean = np.mean(fraction_of_frames[:,b,POST,:],axis=0)
    post_stderr = np.sqrt(np.var(fraction_of_frames[:,b,POST,:],axis=0)/num_of_mice)

    plt.plot(pre_mean, alpha=0.3, color='green', marker='o',label = 'Pre-CNO',lw=1)
    plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3,ls='',lw=1)
    plt.plot(CNO_mean, alpha=0.5, color='red', marker='o',label = 'During-CNO',lw=1)
    plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7,ls='',lw=1)
    plt.plot(post_mean, alpha=0.7, color='green', marker='o',label = 'Post-CNO',lw=1)
    plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7,ls='',lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel('Fraction of frames')
    plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: '+behaviors[b])
    plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
    plt.legend()
    plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2)//2)
    plt.xlabel('Time(m)')
    plt.savefig(output_folder + '/dynamics_V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

plt.figure()
pre_mean = np.mean(clustered_fraction_of_frames[:, VDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,VDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:, VDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:, VDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:, VDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:, VDB, POST, :], axis=0) / num_of_mice)

plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('Fraction of frames')
plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming,body licking, wall licking')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.legend()
plt.xticks(np.arange(0, num_of_bins, 2), np.arange(0, num_of_bins, 2) // 2)
plt.xlabel('Time(m)')
plt.savefig(output_folder + '/dynamics_V_graph_VDB.png', dpi=300)
plt.close()


plt.figure()
pre_mean = np.mean(clustered_fraction_of_frames[:, nVDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,nVDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:, nVDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:, nVDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:, nVDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:, nVDB, POST, :], axis=0) / num_of_mice)

plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('Fraction of frames')
plt.title('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing,BTC,other,jump')
plt.tight_layout(rect=[0.02, 0.01, .95, 0.9], pad=1)
plt.legend()
plt.xticks(np.arange(0, num_of_bins, 2), np.arange(0, num_of_bins, 2) // 2)
plt.xlabel('Time(m)')
plt.savefig(output_folder + '/dynamics_V_graph_nVDB.png', dpi=300)
plt.close()
#%%Changes in distance travelled
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/wo_stat_loco_separation/'
distance_travelled = np.zeros((num_of_mice,num_of_days))
m_idx = 0
for m in sandwich_days:
    distance_travelled[m_idx,PRE] = np.sum(sandwich_days[m]['pre']['locomotion'])
    distance_travelled[m_idx, DURING] = np.sum(sandwich_days[m]['CNO']['locomotion'])
    distance_travelled[m_idx, POST] = np.sum(sandwich_days[m]['post']['locomotion'])

plt.figure()
mean = np.mean(distance_travelled,axis=0)
stderr = np.sqrt(np.var(distance_travelled,axis=0)/num_of_mice)
plt.plot(np.arange(num_of_days),mean,ls='--',marker='o',lw=1,markerfacecolor='white',c='k')
plt.errorbar(np.arange(num_of_days),y=mean,yerr=stderr,color='k',capsize=2,capthick=1,ls='')
plt.title('Distance travelled pre-during-post CNO days')
plt.xlabel('Day')
plt.xticks(np.arange(num_of_days),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('Distance Travelled(m)')
plt.savefig(output_folder+'distance_travelled.png',dpi=300)
plt.close()

#%%States & actions diagram for each day
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/States & Actions/'
transition_prob = np.zeros((num_of_days,num_of_mice,num_of_behaviors, num_of_behaviors))
time_spent=np.zeros((num_of_days,num_of_mice,num_of_behaviors))
mice_mean_transition_prob = np.zeros((num_of_days,num_of_behaviors, num_of_behaviors))
mice_mean_time_spent = np.zeros((num_of_days,num_of_behaviors))
days = ['Pre-CNO','During-CNO','Post-CNO']
m_idx=0
for m in sandwich_days:
    pre = sandwich_days[m]['pre']['merged'][:15*minute]
    CNO = sandwich_days[m]['CNO']['merged'][:15*minute]
    post = sandwich_days[m]['post']['merged'][:15*minute]
    t_idx=0
    for predictions in [pre,CNO,post]:
        for i in range(predictions.size-1):
            transition_prob[t_idx,m_idx,predictions[i],predictions[i+1]]+=1
            time_spent[t_idx,m_idx,predictions[i]]+=1
        time_spent[t_idx, m_idx, predictions[-1]] += 1
        time_spent[t_idx, m_idx, :] /= np.sum(time_spent[t_idx, m_idx, :])
        np.fill_diagonal(transition_prob[t_idx, m_idx, :, :], 0)
        for b in range(num_of_behaviors):
            transition_prob[t_idx,m_idx,b,:]/=np.sum(transition_prob[t_idx,m_idx,b,:])
        mice_mean_transition_prob[t_idx, :, :] = np.sum(transition_prob[t_idx, :, :, :], axis=0) / float(num_of_mice)
        mice_mean_time_spent[t_idx, :] = np.sum(time_spent[t_idx, :, :], axis=0) / float(num_of_mice)
        t_idx+=1
    m_idx+=1

print('Done collecting data')
mice_mean_time_spent*=1000
transition_threshold = 1.0/(num_of_behaviors**2) # randomly selecting behavior and randomly selecting to move to the nexet behavior
aboveChance_alpha=0.8
belowChance_alpha=0.12
for t_idx in range(num_of_days):
    plt.figure()
    edges = []
    edge_colors = []
    edge_width = []
    for b_cur in range(num_of_behaviors):
        for b_next in range(num_of_behaviors):
            edges.append((b_cur,b_next,mice_mean_transition_prob[t_idx,b_cur,b_next]*5))
            if mice_mean_transition_prob[t_idx,b_cur,b_next]>=transition_threshold:
                edge_colors.append([0.294, 0, 0.51, aboveChance_alpha])
            else:
                edge_colors.append([0.294, 0, 0.51, belowChance_alpha])
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges,color='red')
    weights = nx.get_edge_attributes(G, 'weight').values()
    plt.title(days[t_idx]+'\n$\pi$[b$_t$,b$_{t+1}$]')
    nx.draw(G, pos=nx.circular_layout(G),edge_color=edge_colors, node_color=colors,width=list(weights),node_size=mice_mean_time_spent[t_idx],arrowstyle='->'
            ,arrowsize=15 , connectionstyle='arc3, rad = 0.1',with_labels=False)
    plt.savefig(output_folder+'states_actions_'+days[t_idx]+'.png',dpi=300)
    plt.close()


#%% Changes in vigor index
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Movement parameters/'
vigor={}
for m in sandwich_days:
    vigor[m]={}
    vigor_pre = sandwich_days[m]['pre']['motion_index']['cam1']
    preds_pre = sandwich_days[m]['pre']['merged']
    vigor_CNO = sandwich_days[m]['CNO']['motion_index']['cam1']
    preds_CNO = sandwich_days[m]['CNO']['merged']
    vigor_post = sandwich_days[m]['post']['motion_index']['cam1']
    preds_post = sandwich_days[m]['post']['merged']

    for b in range(num_of_behaviors):
        vigor[m][b] = {'pre': vigor_pre[preds_pre==b], 'CNO': vigor_CNO[preds_CNO==b], 'post': vigor_post[preds_post==b]}

for b in range(num_of_behaviors):
    plt.figure()
    x_idx=0
    for m in sandwich_days:
        sns.boxplot(x = [x_idx,x_idx+1,x_idx+2], data =[vigor[m][b]['pre'],vigor[m][b]['CNO'],vigor[m][b]['post']],color=colors[b] )
        plt.xticks(x_idx+1,m)
        x_idx+=3
plt.show()






#%% Changes in number of bouts,bouts length
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Bouts_analysis/'
num_of_bouts = np.zeros((num_of_mice,num_of_days,num_of_behaviors))
num_of_bouts[:,:,:]=np.nan
CDF_bins = np.arange(0,500,0.01)
num_CDF_bins = CDF_bins.size
CDF = np.zeros((num_of_mice,num_of_days,num_of_behaviors,num_CDF_bins))
CDF[:,:,:,:]=np.nan

for b in range(num_of_behaviors):
    m_idx = 0
    for m in sandwich_days:
        predictions_pre = sandwich_days[m]['pre']['merged']
        predictions_CNO = sandwich_days[m]['CNO']['merged']
        predictions_post = sandwich_days[m]['post']['merged']
        print(behaviors[b],m)
        bouts_pre = hf.segment_bouts(predictions_pre, b, 0)
        bouts_CNO = hf.segment_bouts(predictions_CNO, b, 0)
        bouts_post = hf.segment_bouts(predictions_post, b, 0)
        length_pre = np.argsort(bouts_pre['length'])
        length_CNO = np.argsort(bouts_CNO['length'])
        length_post = np.argsort(bouts_post['length'])
        if  np.count_nonzero(predictions_pre == b)>0:
            # if b==3: print('pre',m,behaviors[b],bouts_pre['number'] , np.count_nonzero(predictions_pre == b) , (bouts_pre['number'] / np.count_nonzero(predictions_pre == b)),1-(bouts_pre['number'] / np.count_nonzero(predictions_pre == b)))
            num_of_bouts[m_idx, PRE, b] = 1 - (bouts_pre['number'] / np.count_nonzero(predictions_pre == b))
        else:
            num_of_bouts[m_idx, PRE, b] = 0
        if np.count_nonzero(predictions_CNO == b) > 0:
            # if b == 3: print('CNO',m, behaviors[b], bouts_CNO['number'], np.count_nonzero(predictions_CNO == b),
            #                  (bouts_pre['number'] / np.count_nonzero(predictions_CNO == b)),
            #                  1 - (bouts_pre['number'] / np.count_nonzero(predictions_CNO == b)))
            num_of_bouts[m_idx, DURING, b] = 1 - (bouts_CNO['number'] / np.count_nonzero(predictions_CNO == b))
        else:
            num_of_bouts[m_idx, DURING, b] = 0
        if np.count_nonzero(predictions_post == b)>0:
            num_of_bouts[m_idx, POST, b] = 1 - (bouts_post['number'] / np.count_nonzero(predictions_post == b))
        else:
            num_of_bouts[m_idx, POST, b] = 0
        for bin in range(CDF_bins.size):
            if len(length_pre)>0:
                CDF[m_idx,PRE,b,bin] = np.count_nonzero(length_pre<=CDF_bins[bin])/ length_pre.size
            if len(length_CNO)>0:
                CDF[m_idx, DURING, b, bin] = np.count_nonzero(length_CNO <= CDF_bins[bin]) / length_CNO.size
            if len(length_post)>0:
                CDF[m_idx, POST, b, bin] = np.count_nonzero(length_post <= CDF_bins[bin]) / length_post.size
        m_idx+=1



for b in range(num_of_behaviors):
    fig, ax = plt.subplots(ncols=3, sharey='all')
    for m_idx in range(num_of_mice):
        if not np.isnan(CDF[m_idx,PRE,b,:]).any():
            ax[0].plot(CDF_bins,CDF[m_idx,PRE,b,:].T,color = colors[b],alpha=.5)
    plt.sca(ax[0])
    plt.ylabel('CDF')
    plt.title('Pre-CNO')
    for m_idx in range(num_of_mice):
        if not np.isnan(CDF[m_idx,DURING,b,:]).any():
            ax[1].plot(CDF_bins, CDF[m_idx, DURING, b, :].T, color=colors[b], alpha=.5)
    plt.sca(ax[1])
    plt.xlabel('Bout duration (#frames)')
    plt.title('During-CNO')
    for m_idx in range(num_of_mice):
        if not np.isnan(CDF[m_idx,POST,b,:]).any():
            ax[2].plot(CDF_bins, CDF[m_idx, POST, b, :].T, color=colors[b], alpha=.5)

    plt.sca(ax[2])
    plt.title('Post-CNO')

    plt.suptitle('CDFs of bout duration, behavior : '+behaviors[b])
    plt.savefig(output_folder+'CDF_bout_duration_'+behaviors[b]+'.png',dpi=300)
    plt.close()

    plt.figure()
    plt.plot(np.arange(num_of_days),num_of_bouts[:,:,b].T,color=colors[b])
    plt.xticks(np.arange(num_of_days),['Pre-CNO','During-CNO','Post-CNO'])
    plt.xlabel('Day')
    plt.ylabel('Boutiness (1- $\\frac{|B|}{|F|}$)')
    plt.bar(1, height=1, color='#009FE3', alpha=.3, label='CNO-day')
    plt.title('Boutiness on CNO day, behavior : '+behaviors[b])
    plt.savefig(output_folder+'boutiness_'+behaviors[b]+'.png',dpi=300)
    plt.close()
    # plt.show()


#%% Dynamics of boutiness
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Bouts_analysis/'
T = 25
num_of_bouts = np.zeros((num_of_mice,num_of_days,num_of_behaviors,T))
num_of_bouts[:,:,:,:]=np.nan
noNan_mice =  np.zeros((num_of_days,num_of_behaviors,T))+num_of_mice
for b in range(num_of_behaviors):
    m_idx = 0
    for m in sandwich_days:
        predictions_pre = sandwich_days[m]['pre']['merged']
        predictions_CNO = sandwich_days[m]['CNO']['merged']
        predictions_post = sandwich_days[m]['post']['merged']
        print(behaviors[b],m)
        for t in np.arange(0,T*minute,minute):
            bouts_pre = hf.segment_bouts(predictions_pre[:t], b, 0)
            bouts_CNO = hf.segment_bouts(predictions_CNO[:t], b, 0)
            bouts_post = hf.segment_bouts(predictions_post[:t], b, 0)
            if  np.count_nonzero(predictions_pre[:t] == b)>0:
                num_of_bouts[m_idx, PRE, b,int(t/minute)] = 1 - (bouts_pre['number'] / np.count_nonzero(predictions_pre[:t] == b))
            else:
                num_of_bouts[m_idx, PRE, b,int(t/minute)] = np.nan
                noNan_mice[PRE,b,int(t/minute)]-=1
            if np.count_nonzero(predictions_CNO[:t] == b) > 0:
                num_of_bouts[m_idx, DURING, b,int(t/minute)] = 1 - (bouts_CNO['number'] / np.count_nonzero(predictions_CNO[:t] == b))
            else:
                num_of_bouts[m_idx, DURING, b,int(t/minute)] = np.nan
                noNan_mice[DURING, b, int(t / minute)] -= 1
            if np.count_nonzero(predictions_post[:t] == b)>0:
                num_of_bouts[m_idx, POST, b,int(t/minute)] = 1 - (bouts_post['number'] / np.count_nonzero(predictions_post[:t] == b))
            else:
                num_of_bouts[m_idx, POST, b,int(t/minute)] = np.nan
                noNan_mice[POST, b, int(t / minute)] -= 1
        m_idx+=1



for b in range(num_of_behaviors):

    pre_mean = np.nanmean(num_of_bouts[:,PRE,b,:], axis=0)
    pre_stderr = np.sqrt(np.nanvar(num_of_bouts[:,PRE,b,:],axis=0)/noNan_mice[PRE,b])
    CNO_mean = np.nanmean(num_of_bouts[:,DURING,b,:], axis=0)
    CNO_stderr = np.sqrt(np.nanvar(num_of_bouts[:,DURING,b,:],axis=0)/noNan_mice[DURING,b])
    post_mean = np.nanmean(num_of_bouts[:,POST,b,:], axis=0)
    post_stderr = np.sqrt(np.nanvar(num_of_bouts[:,POST,b,:],axis=0)/noNan_mice[POST,b])
    plt.figure()
    plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
    plt.errorbar(np.arange(T), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
    plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
    plt.errorbar(np.arange(T), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
    plt.errorbar(np.arange(T), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel('Boutiness (1- $\\frac{|B|}{|F|}$)')
    plt.title('Change in boutiness CNO\n Behavior: '+behaviors[b])
    plt.legend()
    plt.xticks(np.arange(T), np.arange(T))
    plt.xlabel('Time(m)')
    plt.savefig(output_folder + '/dynamics_boutiness_'+behaviors[b]+'.png', dpi=300)
    plt.close()




#%% Vigor and velocity histograms around CNO days
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Movement_parameters/'
trials_subset = ['pre','CNO','post']
colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,]
T = 15*minute
vigor={}
velocity = {}
for b in range(num_of_behaviors):
    vigor[b] = {}
    velocity[b] = {}
    for tr in trials_subset:
        vigor[b][tr] = []
        velocity[b][tr] = []
for m in sandwich_days:
    vigor_pre = np.max(np.vstack((sandwich_days[m]['pre']['motion_index']['cam1'],sandwich_days[m]['pre']['motion_index']['cam2'])),axis=0)[:T]
    vel_pre = sandwich_days[m]['pre']['locomotion'][:T]
    preds_pre = sandwich_days[m]['pre']['merged'][:T]
    len_array = np.array([len(sandwich_days[m]['pre']['motion_index']['cam1']), len(sandwich_days[m]['pre']['motion_index']['cam2']),
                          len(sandwich_days[m]['pre']['locomotion']), len(sandwich_days[m]['pre']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: pre Cams are not of the same length"
    vigor_CNO = np.max(np.vstack((sandwich_days[m]['CNO']['motion_index']['cam1'],sandwich_days[m]['CNO']['motion_index']['cam2'])),axis=0)[:T]
    vel_CNO = sandwich_days[m]['CNO']['locomotion'][:T]
    preds_CNO = sandwich_days[m]['CNO']['merged'][:T]

    len_array = np.array([len(sandwich_days[m]['CNO']['motion_index']['cam1']), len(sandwich_days[m]['CNO']['motion_index']['cam2']),
                          len(sandwich_days[m]['CNO']['locomotion']), len(sandwich_days[m]['CNO']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: CNO Cams are not of the same length"

    vigor_post = np.max(np.vstack((sandwich_days[m]['post']['motion_index']['cam1'],sandwich_days[m]['post']['motion_index']['cam2'])),axis=0)[:T]
    vel_post = sandwich_days[m]['post']['locomotion'][:T]
    preds_post = sandwich_days[m]['post']['merged'][:T]

    len_array = np.array([len(sandwich_days[m]['post']['motion_index']['cam1']), len(sandwich_days[m]['post']['motion_index']['cam2']),
                          len(sandwich_days[m]['post']['locomotion']), len(sandwich_days[m]['post']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: post Cams are not of the same length"

    for b in range(num_of_behaviors):
        vigor[b]['pre'].extend(vigor_pre[preds_pre == b])
        velocity[b]['pre'].extend(vel_pre[preds_pre == b])

        vigor[b]['CNO'].extend(vigor_CNO[preds_CNO == b])
        velocity[b]['CNO'].extend(vel_CNO[preds_CNO == b])

        vigor[b]['post'].extend(vigor_post[preds_post == b])
        velocity[b]['post'].extend(vel_post[preds_post == b])


for b in range(num_of_behaviors):
    fig , ax = plt.subplots(ncols = 2,figsize=(14,6))
    plt.sca(ax[0])
    tr_idx = 0
    for tr in trials_subset:
        vigor_b_tr = np.array(vigor[b][tr])
        vigor_b_tr = vigor_b_tr[vigor_b_tr<=150]
        sns.histplot(vigor_b_tr,label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx])
        tr_idx += 1
    plt.xlabel('Vigor (a.u)')
    plt.title('Vigor')
    plt.legend()

    plt.sca(ax[1])
    tr_idx = 0
    for tr in trials_subset:
        vel_b_tr = np.array(velocity[b][tr])*FPS
        vel_b_tr = vel_b_tr[vel_b_tr<=.5]
        sns.histplot(vel_b_tr,label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx])
        tr_idx += 1
    plt.xlabel('Velocity ($\\frac{m}{s}$)')
    plt.title('Velocity')
    plt.legend()

    plt.suptitle('Values around CNO day\n Behavior : '+behaviors[b])
    plt.savefig(output_folder+'/vel_vigor_histograms_'+behaviors[b]+'.png',dpi=300)
    plt.close()

#%% Vigor and velocity histograms around CNO days VDB VS nVDB
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Movement_parameters/'
trials_subset = ['pre','CNO','post']
colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,]
T = 15*minute
vigor={}
velocity = {}
for b in range(num_of_behaviors):
    vigor[b] = {}
    velocity[b] = {}
    for tr in trials_subset:
        vigor[b][tr] = []
        velocity[b][tr] = []
for m in sandwich_days:
    vigor_pre = np.max(np.vstack((sandwich_days[m]['pre']['motion_index']['cam1'],sandwich_days[m]['pre']['motion_index']['cam2'])),axis=0)[:T]
    vel_pre = sandwich_days[m]['pre']['locomotion'][:T]
    preds_pre = sandwich_days[m]['pre']['merged'][:T]
    len_array = np.array([len(sandwich_days[m]['pre']['motion_index']['cam1']), len(sandwich_days[m]['pre']['motion_index']['cam2']),
                          len(sandwich_days[m]['pre']['locomotion']), len(sandwich_days[m]['pre']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: pre Cams are not of the same length"
    vigor_CNO = np.max(np.vstack((sandwich_days[m]['CNO']['motion_index']['cam1'],sandwich_days[m]['CNO']['motion_index']['cam2'])),axis=0)[:T]
    vel_CNO = sandwich_days[m]['CNO']['locomotion'][:T]
    preds_CNO = sandwich_days[m]['CNO']['merged'][:T]

    len_array = np.array([len(sandwich_days[m]['CNO']['motion_index']['cam1']), len(sandwich_days[m]['CNO']['motion_index']['cam2']),
                          len(sandwich_days[m]['CNO']['locomotion']), len(sandwich_days[m]['CNO']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: CNO Cams are not of the same length"

    vigor_post = np.max(np.vstack((sandwich_days[m]['post']['motion_index']['cam1'],sandwich_days[m]['post']['motion_index']['cam2'])),axis=0)[:T]
    vel_post = sandwich_days[m]['post']['locomotion'][:T]
    preds_post = sandwich_days[m]['post']['merged'][:T]

    len_array = np.array([len(sandwich_days[m]['post']['motion_index']['cam1']), len(sandwich_days[m]['post']['motion_index']['cam2']),
                          len(sandwich_days[m]['post']['locomotion']), len(sandwich_days[m]['post']['merged'])])
    assert np.all(len_array == len_array[0]), "ERROR: post Cams are not of the same length"
    clustered_pre = np.where(preds_pre<=3 , VDB , nVDB)
    clustered_CNO = np.where(preds_CNO <= 3, VDB, nVDB)
    clustered_post = np.where(preds_post <= 3, VDB, nVDB)
    for b in [VDB,nVDB]:
        vigor[b]['pre'].extend(vigor_pre[clustered_pre == b])
        velocity[b]['pre'].extend(vel_pre[clustered_pre == b])

        vigor[b]['CNO'].extend(vigor_CNO[clustered_CNO == b])
        velocity[b]['CNO'].extend(vel_CNO[clustered_CNO == b])

        vigor[b]['post'].extend(vigor_post[clustered_post == b])
        velocity[b]['post'].extend(vel_post[clustered_post == b])


for b in [VDB,nVDB]:
    fig , ax = plt.subplots(ncols = 2,figsize=(14,6))
    plt.sca(ax[0])
    tr_idx = 0
    for tr in trials_subset:
        vigor_b_tr = np.array(vigor[b][tr])
        vigor_b_tr = vigor_b_tr[vigor_b_tr<=150]
        sns.histplot(vigor_b_tr,label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx])
        tr_idx += 1
    plt.xlabel('Vigor (a.u)')
    plt.title('Vigor')
    plt.legend()

    plt.sca(ax[1])
    tr_idx = 0
    for tr in trials_subset:
        vel_b_tr = np.array(velocity[b][tr])*FPS
        vel_b_tr = vel_b_tr[vel_b_tr<=.5]
        sns.histplot(vel_b_tr,label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx])
        tr_idx += 1
    plt.xlabel('Velocity ($\\frac{m}{s}$)')
    plt.title('Velocity')
    plt.legend()
    if b == VDB:
        plt.suptitle('Values around CNO day\n Behavior : Grooming, body licking , wall licking , floor licking')
        plt.savefig(output_folder+'/vel_vigor_histograms_VDB.png',dpi=300)
    else:
        plt.suptitle('Values around CNO day\n Behavior : Rearing , other ,  BTC , jump')
        plt.savefig(output_folder + '/vel_vigor_histograms_nVDB.png', dpi=300)
    plt.close()
#%% V graphs  - stationary VS locomotive  - not normalized
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/w_stat_loco_separation/wo_normalization/'

fraction_of_frames = np.zeros((num_of_mice , 2,num_of_behaviors , num_of_days))
clustered_fraction_of_frames = np.zeros((num_of_mice ,2, 2 , num_of_days))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])
    preds_post = np.copy(sandwich_days[m]['post']['merged'])
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])
    for b in range(num_of_behaviors):
        stationary_pre = np.intersect1d(np.nonzero(preds_pre==b)[0] , np.nonzero(vel_pre<=stationary_theta)[0]).size
        locomotive_pre = np.intersect1d(np.nonzero(preds_pre == b)[0], np.nonzero(vel_pre > stationary_theta)[0]).size
        fraction_of_frames[m_idx ,stationary ,  b , PRE] = stationary_pre/preds_pre.size
        fraction_of_frames[m_idx,locomotive, b, PRE] = locomotive_pre / preds_pre.size

        stationary_CNO = np.intersect1d(np.nonzero(preds_CNO==b)[0] , np.nonzero(vel_CNO<=stationary_theta)[0]).size
        locomotive_CNO = np.intersect1d(np.nonzero(preds_CNO == b)[0], np.nonzero(vel_CNO > stationary_theta)[0]).size
        fraction_of_frames[m_idx ,stationary ,  b , DURING] = stationary_CNO/preds_CNO.size
        fraction_of_frames[m_idx,locomotive, b, DURING] = locomotive_CNO / preds_CNO.size

        stationary_post = np.intersect1d(np.nonzero(preds_post==b)[0] , np.nonzero(vel_post<=stationary_theta)[0]).size
        locomotive_post = np.intersect1d(np.nonzero(preds_post == b)[0], np.nonzero(vel_post > stationary_theta)[0]).size
        fraction_of_frames[m_idx ,stationary ,  b , POST] = stationary_post/preds_post.size
        fraction_of_frames[m_idx,locomotive, b, POST] = locomotive_post / preds_post.size


    clustered_pre = np.where(preds_pre<=3,VDB,nVDB)
    clustered_CNO = np.where(preds_CNO<=3,VDB,nVDB)
    clustered_post = np.where(preds_post<=3,VDB,nVDB)

    for category in [VDB,nVDB]:
        stationary_pre = np.intersect1d(np.nonzero(clustered_pre==category)[0] , np.nonzero(vel_pre<=stationary_theta)[0]).size
        locomotive_pre = np.intersect1d(np.nonzero(clustered_pre == category)[0], np.nonzero(vel_pre > stationary_theta)[0]).size
        clustered_fraction_of_frames[m_idx,stationary , category,PRE] = stationary_pre/clustered_pre.size
        clustered_fraction_of_frames[m_idx, locomotive, category, PRE] = locomotive_pre / clustered_pre.size

        stationary_CNO = np.intersect1d(np.nonzero(clustered_CNO==category)[0] , np.nonzero(vel_CNO<=stationary_theta)[0]).size
        locomotive_CNO = np.intersect1d(np.nonzero(clustered_CNO == category)[0], np.nonzero(vel_CNO > stationary_theta)[0]).size
        clustered_fraction_of_frames[m_idx, stationary , category, DURING] = stationary_CNO/ clustered_CNO.size
        clustered_fraction_of_frames[m_idx, locomotive, category, DURING] = locomotive_CNO / clustered_CNO.size

        stationary_post = np.intersect1d(np.nonzero(clustered_post==category)[0] , np.nonzero(vel_post<=stationary_theta)[0]).size
        locomotive_post = np.intersect1d(np.nonzero(clustered_post == category)[0], np.nonzero(vel_post > stationary_theta)[0]).size
        clustered_fraction_of_frames[m_idx, stationary , category, POST] = stationary_post / clustered_post.size
        clustered_fraction_of_frames[m_idx, locomotive, category, POST] = locomotive_post / clustered_post.size
    m_idx+=1


print(fraction_of_frames)
for b in range(num_of_behaviors):
    fig , ax = plt.subplots (ncols=2,sharey='all')
    plt.sca(ax[0])
    mean = np.mean(fraction_of_frames[:,stationary , b,:],axis=0)
    stderr = np.sqrt(np.var(fraction_of_frames[:, stationary , b, :], axis=0)/num_of_mice)
    plt.plot(fraction_of_frames[:,stationary , b,:].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean.T, color=colors[b])
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylabel('Fraction of frames')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Stationary')
    plt.sca(ax[1])
    mean = np.mean(fraction_of_frames[:,locomotive , b,:],axis=0)
    stderr = np.sqrt(np.var(fraction_of_frames[:, locomotive , b, :], axis=0)/num_of_mice)
    plt.plot(fraction_of_frames[:,locomotive , b,:].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean.T, color=colors[b])
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Locomotive')
    plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: '+behaviors[b])
    plt.subplots_adjust(top=.85)
    plt.savefig(output_folder + '/V_graph_statVSloco_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

fig , ax = plt.subplots (ncols=2,sharey='all')
plt.sca(ax[0])
mean = np.mean(clustered_fraction_of_frames[:,stationary , VDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary , VDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,stationary , VDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('Fraction of frames')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Stationary')
plt.sca(ax[1])
mean = np.mean(clustered_fraction_of_frames[:,locomotive , VDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive , VDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,locomotive , VDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Locomotive')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming, body licking , floor licking , wall licking')
plt.subplots_adjust( top=.85)
plt.savefig(output_folder + '/V_graph_statVSloco_VDB.png', dpi=300)
plt.close()

fig , ax = plt.subplots (ncols=2,sharey='all')
plt.sca(ax[0])
mean = np.mean(clustered_fraction_of_frames[:,stationary , nVDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary , nVDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,stationary , nVDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('Fraction of frames')
plt.yticks([0,0.5,1],[0,0.5,1])
plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Stationary')
plt.sca(ax[1])
mean = np.mean(clustered_fraction_of_frames[:,locomotive , VDB,:],axis=0)
stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive , VDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,locomotive , VDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Locomotive')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing , other , BTC, jump')
plt.subplots_adjust( top=.85)
plt.savefig(output_folder + '/V_graph_statVSloco_nVDB.png', dpi=300)
plt.close()


#%% Dynamics of CNO effect on stereotypies  - stationary VS locomotive - all behaviors - Not normalised
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/w_stat_loco_separation/wo_normalization/'
num_of_bins = 25
cutoff = 25*minute
fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_behaviors , num_of_days,num_of_bins))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])[:cutoff]
    edges_pre = np.linspace(0,preds_pre.size,num_of_bins+1,dtype=int)
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])[:cutoff]
    edges_CNO = np.linspace(0, preds_CNO.size, num_of_bins+1, dtype=int)
    preds_post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])[:cutoff]
    edges_post = np.linspace(0, preds_post.size, num_of_bins+1, dtype=int)

    for b in range(num_of_behaviors):
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end-bin_start
            stationary_pre = np.intersect1d(np.nonzero(preds_pre[bin_start:bin_end]==b)[0],np.nonzero(vel_pre[bin_start:bin_end]<=stationary_theta)[0]).size
            locomotive_pre = np.intersect1d(np.nonzero(preds_pre[bin_start:bin_end]==b)[0],np.nonzero(vel_pre[bin_start:bin_end]>stationary_theta)[0]).size
            fraction_of_frames[m_idx ,stationary, b , PRE,edge] = stationary_pre/bin_size
            fraction_of_frames[m_idx, locomotive, b, PRE, edge] = locomotive_pre / bin_size

            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge + 1]
            bin_size = bin_end - bin_start
            stationary_CNO = np.intersect1d(np.nonzero(preds_CNO[bin_start:bin_end] == b)[0],np.nonzero(vel_CNO[bin_start:bin_end] <= stationary_theta)[0]).size
            locomotive_CNO = np.intersect1d(np.nonzero(preds_CNO[bin_start:bin_end] == b)[0],np.nonzero(vel_CNO[bin_start:bin_end] > stationary_theta)[0]).size
            fraction_of_frames[m_idx, stationary, b, DURING, edge] = stationary_CNO / bin_size
            fraction_of_frames[m_idx, locomotive, b, DURING, edge] = locomotive_CNO / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            stationary_post = np.intersect1d(np.nonzero(preds_post[bin_start:bin_end] == b)[0],np.nonzero(vel_post[bin_start:bin_end] <= stationary_theta)[0]).size
            locomotive_post = np.intersect1d(np.nonzero(preds_post[bin_start:bin_end] == b)[0],np.nonzero(vel_post[bin_start:bin_end] > stationary_theta)[0]).size
            fraction_of_frames[m_idx, stationary, b, POST, edge] = stationary_post / bin_size
            fraction_of_frames[m_idx, locomotive, b, POST, edge] = locomotive_post / bin_size
    m_idx+=1

for b in range(num_of_behaviors):
    fig, ax = plt.subplots(ncols=2, sharey='all', figsize=(12, 5))
    plt.sca(ax[0])
    pre_mean = np.mean(fraction_of_frames[:, stationary, b, PRE, :], axis=0)
    pre_stderr = np.sqrt(np.var(fraction_of_frames[:, stationary, b, PRE, :], axis=0) / num_of_mice)
    CNO_mean = np.mean(fraction_of_frames[:, stationary, b, DURING, :], axis=0)
    CNO_stderr = np.sqrt(np.var(fraction_of_frames[:, stationary, b, DURING, :], axis=0) / num_of_mice)
    post_mean = np.mean(fraction_of_frames[:, stationary, b, POST, :], axis=0)
    post_stderr = np.sqrt(np.var(fraction_of_frames[:, stationary, b, POST, :], axis=0) / num_of_mice)

    plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
    plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel('Fraction of frames')
    plt.title('Stationary')
    plt.legend()
    plt.xticks(np.arange(0, num_of_bins, 2), np.arange(0, num_of_bins, 2))
    plt.xlabel('Time(m)')
    plt.sca(ax[1])
    pre_mean = np.mean(fraction_of_frames[:, locomotive, b, PRE, :], axis=0)
    pre_stderr = np.sqrt(np.var(fraction_of_frames[:, locomotive, b, PRE, :], axis=0) / num_of_mice)
    CNO_mean = np.mean(fraction_of_frames[:, locomotive, b, DURING, :], axis=0)
    CNO_stderr = np.sqrt(np.var(fraction_of_frames[:, locomotive, b, DURING, :], axis=0) / num_of_mice)
    post_mean = np.mean(fraction_of_frames[:, locomotive, b, POST, :], axis=0)
    post_stderr = np.sqrt(np.var(fraction_of_frames[:, locomotive, b, POST, :], axis=0) / num_of_mice)

    plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
    plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Locomotive')
    plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: ' + behaviors[b])
    plt.legend()
    plt.xticks(np.arange(0, num_of_bins, 2), np.arange(0, num_of_bins, 2))
    plt.xlabel('Time(m)')
    plt.subplots_adjust(top=0.85, bottom=0.1)
    plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_' + behaviors[b] + '.png', dpi=300)
    plt.close()


#%% Dynamics of CNO effect on stereotypies  - stationary VS locomotive - VDB/nVDB - Not normalized
output_folder = 'Figures/'+model+'/DREADDS/' + cohort + '/Summary/w_stat_loco_separation/wo_normalization/'
num_of_bins = 25
cutoff = 25 * minute
clustered_fraction_of_frames = np.zeros((num_of_mice, 2, 2, num_of_days, num_of_bins))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])[:cutoff]
    edges_pre = np.linspace(0, preds_pre.size, num_of_bins + 1, dtype=int)
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])[:cutoff]
    edges_CNO = np.linspace(0, preds_CNO.size, num_of_bins + 1, dtype=int)
    preds_post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])[:cutoff]
    edges_post = np.linspace(0, preds_post.size, num_of_bins + 1, dtype=int)

    clustered_pre = np.where(preds_pre<=3,VDB,nVDB)
    clustered_CNO = np.where(preds_CNO<=3,VDB,nVDB)
    clustered_post = np.where(preds_post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge+1]
            bin_size = bin_end - bin_start
            stationary_pre = np.intersect1d(np.nonzero(clustered_pre[bin_start:bin_end]==category)[0],np.nonzero(vel_pre[bin_start:bin_end]<=stationary_theta)[0]).size
            locomotive_pre = np.intersect1d(np.nonzero(clustered_pre[bin_start:bin_end]==category)[0],np.nonzero(vel_pre[bin_start:bin_end]>stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx,stationary , category,PRE,edge] = stationary_pre/bin_size
            clustered_fraction_of_frames[m_idx, locomotive, category, PRE, edge] = locomotive_pre / bin_size
            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge+1]
            bin_size = bin_end - bin_start
            stationary_CNO = np.intersect1d(np.nonzero(clustered_CNO[bin_start:bin_end] == category)[0],np.nonzero(vel_CNO[bin_start:bin_end] <= stationary_theta)[0]).size
            locomotive_CNO = np.intersect1d(np.nonzero(clustered_CNO[bin_start:bin_end] == category)[0],np.nonzero(vel_CNO[bin_start:bin_end] > stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx, stationary, category, DURING, edge] = stationary_CNO / bin_size
            clustered_fraction_of_frames[m_idx, locomotive, category, DURING, edge] = locomotive_CNO / bin_size
            bin_start = edges_post[edge]
            bin_end = edges_post[edge+1]
            bin_size = bin_end - bin_start
            stationary_post = np.intersect1d(np.nonzero(clustered_post[bin_start:bin_end] == category)[0],np.nonzero(vel_post[bin_start:bin_end] <= stationary_theta)[0]).size
            locomotive_post = np.intersect1d(np.nonzero(clustered_post[bin_start:bin_end] == category)[0],np.nonzero(vel_post[bin_start:bin_end] > stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx, stationary, category, POST, edge] = stationary_post / bin_size
            clustered_fraction_of_frames[m_idx, locomotive, category, POST, edge] = locomotive_post / bin_size
    m_idx+=1



fig ,ax = plt.subplots(ncols=2 , sharey = 'all',figsize=(12, 5))
pre_mean = np.mean(clustered_fraction_of_frames[:,stationary, VDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary,VDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:,stationary, VDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary, VDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:,stationary, VDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary, VDB, POST, :], axis=0) / num_of_mice)
plt.sca(ax[0])
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('Fraction of frames')
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Stationary')
plt.sca(ax[1])
pre_mean = np.mean(clustered_fraction_of_frames[:,locomotive, VDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive,VDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:,locomotive, VDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive, VDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:,locomotive, VDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive, VDB, POST, :], axis=0) / num_of_mice)
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Locomotive')
plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Grooming,body licking, wall licking')
plt.subplots_adjust(top = 0.85,bottom = 0.1)
plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_VDB.png', dpi=300)
plt.close()


fig ,ax = plt.subplots(ncols=2 , sharey = 'all',figsize=(12, 5))
pre_mean = np.mean(clustered_fraction_of_frames[:,stationary, nVDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary,nVDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:,stationary, nVDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary, nVDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:,stationary, nVDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,stationary, nVDB, POST, :], axis=0) / num_of_mice)
plt.sca(ax[0])
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('Fraction of frames')
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Stationary')
plt.sca(ax[1])
pre_mean = np.mean(clustered_fraction_of_frames[:,locomotive, nVDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive,nVDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.mean(clustered_fraction_of_frames[:,locomotive, nVDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive, nVDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.mean(clustered_fraction_of_frames[:,locomotive, nVDB, POST, :], axis=0)
post_stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,locomotive, nVDB, POST, :], axis=0) / num_of_mice)
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Locomotive')

plt.suptitle('Change in behavioral landscape before-during-after CNO\n Behaviors: Rearing , other , BTC , jump')
plt.subplots_adjust(top = 0.85,bottom = 0.1)
plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_nVDB.png', dpi=300)
plt.close()

#%% V graphs - stationary VS locomotive  - normalized to the number of frames
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/w_stat_loco_separation/w_normalization/'
fraction_of_frames = np.zeros((num_of_mice , 2,num_of_behaviors , num_of_days))
clustered_fraction_of_frames = np.zeros((num_of_mice ,2, 2 , num_of_days))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])
    preds_post = np.copy(sandwich_days[m]['post']['merged'])
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])
    for b in range(num_of_behaviors):
        if np.count_nonzero(preds_pre==b)>0:
            stationary_pre = np.intersect1d(np.nonzero(preds_pre==b)[0] , np.nonzero(vel_pre<=stationary_theta)[0]).size
            fraction_of_frames[m_idx ,stationary ,  b , PRE] = stationary_pre/np.count_nonzero(preds_pre==b)
        if np.count_nonzero(preds_CNO==b)>0:
            stationary_CNO = np.intersect1d(np.nonzero(preds_CNO==b)[0] , np.nonzero(vel_CNO<=stationary_theta)[0]).size
            fraction_of_frames[m_idx ,stationary ,  b , DURING] = stationary_CNO/np.count_nonzero(preds_CNO==b)

        if np.count_nonzero(preds_post==b)>0:
            stationary_post = np.intersect1d(np.nonzero(preds_post==b)[0] , np.nonzero(vel_post<=stationary_theta)[0]).size
            fraction_of_frames[m_idx ,stationary ,  b , POST] = stationary_post/np.count_nonzero(preds_post==b)



    clustered_pre = np.where(preds_pre<=3,VDB,nVDB)
    clustered_CNO = np.where(preds_CNO<=3,VDB,nVDB)
    clustered_post = np.where(preds_post<=3,VDB,nVDB)

    for category in [VDB,nVDB]:
        if np.count_nonzero(clustered_pre==category)>0:
            stationary_pre = np.intersect1d(np.nonzero(clustered_pre==category)[0] , np.nonzero(vel_pre<=stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx,stationary , category,PRE] = stationary_pre/np.count_nonzero(clustered_pre==category)
        else:
            clustered_fraction_of_frames[m_idx, stationary, category, PRE] = np.nan
        if np.count_nonzero(clustered_CNO==category)>0:
            stationary_CNO = np.intersect1d(np.nonzero(clustered_CNO==category)[0] , np.nonzero(vel_CNO<=stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx, stationary , category, DURING] = stationary_CNO/ np.count_nonzero(clustered_CNO==category)
        else:
            clustered_fraction_of_frames[m_idx, stationary, category, DURING] = np.nan
        if np.count_nonzero(clustered_post==category)>0:
            stationary_post = np.intersect1d(np.nonzero(clustered_post==category)[0] , np.nonzero(vel_post<=stationary_theta)[0]).size
            clustered_fraction_of_frames[m_idx, stationary , category, POST] = stationary_post / np.count_nonzero(clustered_post==category)
        else:
            clustered_fraction_of_frames[m_idx, stationary, category, POST] = np.nan
    m_idx+=1


print(fraction_of_frames)
for b in range(num_of_behaviors):
    plt.figure()

    mean = np.nanmean(fraction_of_frames[:,stationary , b,:],axis=0)
    stderr = np.sqrt(np.nanvar(fraction_of_frames[:, stationary , b, :], axis=0)/num_of_mice)
    plt.plot(fraction_of_frames[:,stationary , b,:].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean.T, color=colors[b])
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylabel('$\\frac{|S|}{|F|}$')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Relative fraction of stationary frames out of '+behaviors[b]+' frames\n Behavior: '+behaviors[b])
    plt.savefig(output_folder + '/V_graph_statVSloco_normelized2F'+behaviors[b]+ '.png', dpi=300)
    plt.close()

plt.figure()
mean = np.nanmean(clustered_fraction_of_frames[:,stationary , VDB,:],axis=0)
stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary , VDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,stationary , VDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('$\\frac{|S|}{|F|}$')
# plt.yticks([0,0.5,1],[0,0.5,1])
# plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Relative fraction of stationary frames out of VDB frames\n Behaviors: Grooming , body licking , wall licking , floor licking')
plt.savefig(output_folder + '/V_graph_statVSloco_normelized2F_VDB.png', dpi=300)
plt.close()

plt.figure()
mean = np.nanmean(clustered_fraction_of_frames[:,stationary , nVDB,:],axis=0)
stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary , nVDB,:], axis=0)/num_of_mice)
plt.plot(clustered_fraction_of_frames[:,stationary , nVDB,:].T, alpha=0.3, color='gray', marker='o')
plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
plt.plot(mean, color='#730AFF')
plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color='#730AFF',capsize=2,capthick=1)
plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
plt.ylabel('$\\frac{|S|}{|F|}$')
# plt.yticks([0,0.5,1],[0,0.5,1])
# plt.ylim([0,1])
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.title('Relative fraction of stationary frames out of nVDB frames\n Behaviors: Rearing , other , BTC , jump')
plt.savefig(output_folder + '/V_graph_statVSloco_normelized2F_nVDB.png', dpi=300)
plt.close()

#%% Dynamics of CNO effect on stereotypies  - stationary VS locomotive - all behaviors - Normalised
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Summary/w_stat_loco_separation/w_normalization/'
num_of_bins = 25
cutoff = 25*minute
fraction_of_frames = np.zeros((num_of_mice , 2 , num_of_behaviors , num_of_days,num_of_bins))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])[:cutoff]
    edges_pre = np.linspace(0,preds_pre.size,num_of_bins+1,dtype=int)
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])[:cutoff]
    edges_CNO = np.linspace(0, preds_CNO.size, num_of_bins+1, dtype=int)
    preds_post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])[:cutoff]
    edges_post = np.linspace(0, preds_post.size, num_of_bins+1, dtype=int)

    for b in range(num_of_behaviors):
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge + 1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(preds_pre[bin_start:bin_end]==b)>0:
                stationary_pre = np.intersect1d(np.nonzero(preds_pre[bin_start:bin_end]==b)[0],np.nonzero(vel_pre[bin_start:bin_end]<=stationary_theta)[0]).size
                fraction_of_frames[m_idx ,stationary, b , PRE,edge] = stationary_pre/np.count_nonzero(preds_pre[bin_start:bin_end]==b)
            else:
                fraction_of_frames[m_idx, stationary, b, PRE, edge] = np.nan

            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge + 1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(preds_CNO[bin_start:bin_end] == b) >0:
                stationary_CNO = np.intersect1d(np.nonzero(preds_CNO[bin_start:bin_end] == b)[0],np.nonzero(vel_CNO[bin_start:bin_end] <= stationary_theta)[0]).size
                fraction_of_frames[m_idx, stationary, b, DURING, edge] = stationary_CNO / np.count_nonzero(preds_CNO[bin_start:bin_end] == b)
            else:
                fraction_of_frames[m_idx, stationary, b, DURING, edge] = np.nan
            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(preds_post[bin_start:bin_end] == b)>0:
                stationary_post = np.intersect1d(np.nonzero(preds_post[bin_start:bin_end] == b)[0],np.nonzero(vel_post[bin_start:bin_end] <= stationary_theta)[0]).size
                fraction_of_frames[m_idx, stationary, b, POST, edge] = stationary_post / np.count_nonzero(preds_post[bin_start:bin_end] == b)
            else:
                fraction_of_frames[m_idx, stationary, b, POST, edge] = np.nan
    m_idx+=1


for b in range(num_of_behaviors):
    plt.figure(figsize=(12, 5))

    pre_mean = np.nanmean(fraction_of_frames[:, stationary, b, PRE, :], axis=0)
    pre_stderr = np.sqrt(np.nanvar(fraction_of_frames[:, stationary, b, PRE, :], axis=0) / num_of_mice)
    CNO_mean = np.nanmean(fraction_of_frames[:, stationary, b, DURING, :], axis=0)
    CNO_stderr = np.sqrt(np.nanvar(fraction_of_frames[:, stationary, b, DURING, :], axis=0) / num_of_mice)
    post_mean = np.nanmean(fraction_of_frames[:, stationary, b, POST, :], axis=0)
    post_stderr = np.sqrt(np.nanvar(fraction_of_frames[:, stationary, b, POST, :], axis=0) / num_of_mice)

    plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
    plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
    plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.yticks([0, 0.5, 1], [0, 0.5, 1])
    plt.ylim([0, 1])
    plt.ylabel('$\\frac{|S|}{|F|}$')
    plt.legend()
    plt.xticks(np.arange(0, num_of_bins, 2), np.arange(0, num_of_bins, 2))
    plt.xlabel('Time(m)')
    plt.title('Dynamics of stationary/#frames ratio before-during-after CNO\n Behaviors: ' + behaviors[b])
    # plt.subplots_adjust(top=0.85, bottom=0.1)
    plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_' + behaviors[b] + '_normalized.png', dpi=300)
    plt.close()


#%% Dynamics of CNO effect on stereotypies  - stationary VS locomotive - VDB/nVDB - normalized
output_folder = 'Figures/'+model+'/DREADDS/' + cohort + '/Summary/w_stat_loco_separation/w_normalization/'
num_of_bins = 25
cutoff = 25 * minute
clustered_fraction_of_frames = np.zeros((num_of_mice, 2, 2, num_of_days, num_of_bins))
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])[:cutoff]
    edges_pre = np.linspace(0, preds_pre.size, num_of_bins + 1, dtype=int)
    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])[:cutoff]
    edges_CNO = np.linspace(0, preds_CNO.size, num_of_bins + 1, dtype=int)
    preds_post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])[:cutoff]
    edges_post = np.linspace(0, preds_post.size, num_of_bins + 1, dtype=int)

    clustered_pre = np.where(preds_pre<=3,VDB,nVDB)
    clustered_CNO = np.where(preds_CNO<=3,VDB,nVDB)
    clustered_post = np.where(preds_post<=3,VDB,nVDB)
    for category in [VDB,nVDB]:
        for edge in range(num_of_bins):
            bin_start = edges_pre[edge]
            bin_end = edges_pre[edge + 1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(clustered_pre[bin_start:bin_end] == category)>0:
                stationary_pre = np.intersect1d(np.nonzero(clustered_pre[bin_start:bin_end]==category)[0],np.nonzero(vel_pre[bin_start:bin_end]<=stationary_theta)[0]).size
                clustered_fraction_of_frames[m_idx,stationary , category,PRE,edge] = stationary_pre/np.count_nonzero(clustered_pre[bin_start:bin_end]==category)
            else:
                clustered_fraction_of_frames[m_idx, stationary, category, PRE, edge] = np.nan

            bin_start = edges_CNO[edge]
            bin_end = edges_CNO[edge+1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(clustered_CNO[bin_start:bin_end] == category)>0:

                stationary_CNO = np.intersect1d(np.nonzero(clustered_CNO[bin_start:bin_end] == category)[0],np.nonzero(vel_CNO[bin_start:bin_end] <= stationary_theta)[0]).size
                clustered_fraction_of_frames[m_idx, stationary, category, DURING, edge] = stationary_CNO / np.count_nonzero(clustered_CNO[bin_start:bin_end] == category)
            else:
                clustered_fraction_of_frames[m_idx, stationary, category, DURING, edge] = np.nan

            bin_start = edges_post[edge]
            bin_end = edges_post[edge + 1]
            bin_size = bin_end - bin_start
            if np.count_nonzero(clustered_post[bin_start:bin_end] == category)>0:

                stationary_post = np.intersect1d(np.nonzero(clustered_post[bin_start:bin_end] == category)[0],np.nonzero(vel_post[bin_start:bin_end] <= stationary_theta)[0]).size
                clustered_fraction_of_frames[m_idx, stationary, category, POST, edge] = stationary_post / np.count_nonzero(clustered_post[bin_start:bin_end] == category)
            else:
                clustered_fraction_of_frames[m_idx, stationary, category, POST, edge] = np.nan
    m_idx+=1



plt.figure(figsize=(12, 5))
pre_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, VDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary,VDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, VDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary, VDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, VDB, POST, :], axis=0)
post_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary, VDB, POST, :], axis=0) / num_of_mice)
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('$\\frac{|S|}{|F|}$')
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Dynamics of stationary/#frames ratio before-during-after CNO\n Behaviors: Grooming, body licking , wall licking , wall licking')
plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_VDB_normalized.png', dpi=300)
plt.close()



plt.figure(figsize=(12, 5))
pre_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, nVDB, PRE, :], axis=0)
pre_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary,nVDB, PRE, :], axis=0) / num_of_mice)
CNO_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, nVDB, DURING, :], axis=0)
CNO_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary, nVDB, DURING, :], axis=0) / num_of_mice)
post_mean = np.nanmean(clustered_fraction_of_frames[:,stationary, nVDB, POST, :], axis=0)
post_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[:,stationary, nVDB, POST, :], axis=0) / num_of_mice)
plt.plot(pre_mean, alpha=0.3, color='green', marker='o', label='Pre-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=pre_mean, yerr=pre_stderr, color='green', capsize=2, capthick=.5, alpha=0.3, ls='', lw=1)
plt.plot(CNO_mean, alpha=0.5, color='red', marker='o', label='During-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=CNO_mean, yerr=CNO_stderr, color='red', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
plt.plot(post_mean, alpha=0.7, color='green', marker='o', label='Post-CNO', lw=1)
plt.errorbar(np.arange(num_of_bins), y=post_mean, yerr=post_stderr, color='green', capsize=2, capthick=.5, alpha=0.7, ls='', lw=1)
axes = plt.gca()
axes.spines['top'].set_visible(False)
axes.spines['right'].set_visible(False)
plt.yticks([0, 0.5, 1], [0, 0.5, 1])
plt.ylim([0, 1])
plt.ylabel('$\\frac{|S|}{|F|}$')
plt.legend()
plt.xticks(np.arange(0,num_of_bins,2),np.arange(0,num_of_bins,2))
plt.xlabel('Time(m)')
plt.title('Dynamics of stationary/#frames ratio before-during-after CNO\n Behaviors: Rearing , other , BTC , jump')
# plt.subplots_adjust(top = 0.85,bottom = 0.1)
plt.savefig(output_folder + '/dynamics_V_graph_statVSloco_nVDB_normalized.png', dpi=300)
plt.close()
#%% Comparing vigor values for stationary vs locomotive actions
output_folder = 'Figures/'+model+'/DREADDS/'+cohort+'/Movement_parameters/w_stat_loco_separation/'
vigor = {}
clustered_vigor = {}
for day in ['pre','CNO','post']:
    vigor[day] = {}
    for b in range(num_of_behaviors):
        vigor[day][b] = {'stationary':[] , 'locomotive':[]}
for day in ['pre','CNO','post']:
    clustered_vigor[day] = {}
    for category in [VDB , nVDB]:
        clustered_vigor[day][category] = {'stationary':[] , 'locomotive':[]}
m_idx = 0
for m in sandwich_days:
    preds_pre = np.copy(sandwich_days[m]['pre']['merged'])
    vel_pre = np.copy(sandwich_days[m]['pre']['locomotion'])
    delta = sandwich_days[m]['pre']['motion_index']['cam1'].size - sandwich_days[m]['pre']['motion_index']['cam2'].size
    if delta >0:
        sandwich_days[m]['pre']['motion_index']['cam1'] = sandwich_days[m]['pre']['motion_index']['cam1'][delta:]
    elif delta <0:
        sandwich_days[m]['pre']['motion_index']['cam2'] = sandwich_days[m]['pre']['motion_index']['cam2'][delta:]
    vigor_pre = np.max(np.vstack([sandwich_days[m]['pre']['motion_index']['cam1'],sandwich_days[m]['pre']['motion_index']['cam2']]),axis=0)

    preds_CNO = np.copy(sandwich_days[m]['CNO']['merged'])
    vel_CNO = np.copy(sandwich_days[m]['CNO']['locomotion'])
    delta = sandwich_days[m]['CNO']['motion_index']['cam1'].size - sandwich_days[m]['CNO']['motion_index']['cam2'].size
    if delta > 0:
        sandwich_days[m]['CNO']['motion_index']['cam1'] = sandwich_days[m]['CNO']['motion_index']['cam1'][delta:]
    elif delta < 0:
        sandwich_days[m]['CNO']['motion_index']['cam2'] = sandwich_days[m]['CNO']['motion_index']['cam2'][delta:]
    vigor_CNO = np.max(np.vstack([sandwich_days[m]['CNO']['motion_index']['cam1'], sandwich_days[m]['CNO']['motion_index']['cam2']]), axis=0)

    preds_post = np.copy(sandwich_days[m]['post']['merged'])
    vel_post = np.copy(sandwich_days[m]['post']['locomotion'])
    delta = sandwich_days[m]['post']['motion_index']['cam1'].size - sandwich_days[m]['post']['motion_index']['cam2'].size
    if delta > 0:
        sandwich_days[m]['post']['motion_index']['cam1'] = sandwich_days[m]['post']['motion_index']['cam1'][delta:]
    elif delta < 0:
        sandwich_days[m]['post']['motion_index']['cam2'] = sandwich_days[m]['post']['motion_index']['cam2'][delta:]
    vigor_post = np.max(np.vstack([sandwich_days[m]['post']['motion_index']['cam1'], sandwich_days[m]['post']['motion_index']['cam2']]), axis=0)
    for b in range(num_of_behaviors):
        stat_b = np.intersect1d(np.nonzero(preds_pre==b)[0],np.nonzero(vel_pre<=stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(preds_pre==b)[0],np.nonzero(vel_pre>stationary_theta)[0])
        vigor['pre'][b]['stationary'].extend(vigor_pre[stat_b])
        vigor['pre'][b]['locomotive'].extend(vigor_pre[loco_b])

        stat_b = np.intersect1d(np.nonzero(preds_CNO == b)[0], np.nonzero(vel_CNO <= stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(preds_CNO == b)[0], np.nonzero(vel_CNO > stationary_theta)[0])
        vigor['CNO'][b]['stationary'].extend(vigor_CNO[stat_b])
        vigor['CNO'][b]['locomotive'].extend(vigor_CNO[loco_b])

        stat_b = np.intersect1d(np.nonzero(preds_post == b)[0], np.nonzero(vel_post <= stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(preds_post == b)[0], np.nonzero(vel_post > stationary_theta)[0])
        vigor['post'][b]['stationary'].extend(vigor_post[stat_b])
        vigor['post'][b]['locomotive'].extend(vigor_post[loco_b])

    clustered_pre = np.where(preds_pre<=3,VDB,nVDB)
    clustered_CNO = np.where(preds_CNO<=3,VDB,nVDB)
    clustered_post = np.where(preds_post<=3,VDB,nVDB)

    for category in [VDB,nVDB]:
        stat_b = np.intersect1d(np.nonzero(clustered_pre==category)[0],np.nonzero(vel_pre<=stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(clustered_pre==category)[0],np.nonzero(vel_pre>stationary_theta)[0])
        clustered_vigor['pre'][category]['stationary'].extend(vigor_pre[stat_b])
        clustered_vigor['pre'][category]['locomotive'].extend(vigor_pre[loco_b])

        stat_b = np.intersect1d(np.nonzero(clustered_CNO == category)[0], np.nonzero(vel_CNO <= stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(clustered_CNO == category)[0], np.nonzero(vel_CNO > stationary_theta)[0])
        clustered_vigor['CNO'][category]['stationary'].extend(vigor_CNO[stat_b])
        clustered_vigor['CNO'][category]['locomotive'].extend(vigor_CNO[loco_b])

        stat_b = np.intersect1d(np.nonzero(clustered_post == category)[0], np.nonzero(vel_post <= stationary_theta)[0])
        loco_b = np.intersect1d(np.nonzero(clustered_post == category)[0], np.nonzero(vel_post > stationary_theta)[0])
        clustered_vigor['post'][category]['stationary'].extend(vigor_post[stat_b])
        clustered_vigor['post'][category]['locomotive'].extend(vigor_post[loco_b])
    m_idx+=1

colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,]
for b in range(num_of_behaviors):
    fig,ax = plt.subplots(ncols=2, figsize=(14, 6))
    plt.sca(ax[0])
    sns.histplot(vigor['pre'][b]['stationary'],stat='probability', color='gray',alpha = .3,label = 'Pre')
    sns.histplot(vigor['CNO'][b]['stationary'],stat='probability', color='blue',alpha = .5,label = 'CNO')
    sns.histplot(vigor['post'][b]['stationary'],stat='probability', color='gray',alpha = .7,label = 'Post')
    plt.xlabel('Vigor (a.u)')
    plt.title('Stationary')
    plt.legend()

    plt.sca(ax[1])
    sns.histplot(vigor['pre'][b]['locomotive'],stat='probability', color='gray',alpha = .3,label = 'Pre')
    sns.histplot(vigor['CNO'][b]['locomotive'],stat='probability', color='blue',alpha = .5,label = 'CNO')
    sns.histplot(vigor['post'][b]['locomotive'],stat='probability', color='gray',alpha = .7,label = 'Post')
    plt.xlabel('Vigor (a.u)')
    plt.title('Locomotive')
    plt.legend()

    plt.suptitle('Vigor of stationary and locomotive actions around CNO day\n Behavior : ' + behaviors[b])
    plt.savefig(output_folder + '/stat_loco_vigor_histograms_' + behaviors[b] + '.png', dpi=300)
    plt.close()


fig,ax = plt.subplots(ncols=2, figsize=(14, 6))
plt.sca(ax[0])
sns.histplot(clustered_vigor['pre'][VDB]['stationary'],stat='probability', color='gray',alpha = .3,label = 'Pre')
sns.histplot(clustered_vigor['CNO'][VDB]['stationary'],stat='probability', color='blue',alpha = .5,label = 'CNO')
sns.histplot(clustered_vigor['post'][VDB]['stationary'],stat='probability', color='gray',alpha = .7,label = 'Post')
plt.xlabel('Vigor (a.u)')
plt.title('Stationary')
plt.legend()

plt.sca(ax[1])
sns.histplot(clustered_vigor['pre'][VDB]['locomotive'],stat='probability', color='gray',alpha = .3,label = 'Pre')
sns.histplot(clustered_vigor['CNO'][VDB]['locomotive'],stat='probability', color='blue',alpha = .5,label = 'CNO')
sns.histplot(clustered_vigor['post'][VDB]['locomotive'],stat='probability', color='gray',alpha = .7,label = 'Post')
plt.xlabel('Vigor (a.u)')
plt.title('Locomotive')
plt.legend()

plt.suptitle('Vigor of stationary and locomotive actions around CNO day\n Behaviors : Grooming , body licking , wall licking , floor licking')
plt.savefig(output_folder + '/stat_loco_vigor_histograms_VDB.png', dpi=300)
plt.close()



fig,ax = plt.subplots(ncols=2, figsize=(14, 6))
plt.sca(ax[0])
sns.histplot(clustered_vigor['pre'][nVDB]['stationary'],stat='probability', color='gray',alpha = .3,label = 'Pre')
sns.histplot(clustered_vigor['CNO'][nVDB]['stationary'],stat='probability', color='blue',alpha = .5,label = 'CNO')
sns.histplot(clustered_vigor['post'][nVDB]['stationary'],stat='probability', color='gray',alpha = .7,label = 'Post')
plt.xlabel('Vigor (a.u)')
plt.title('Stationary')
plt.legend()

plt.sca(ax[1])
sns.histplot(clustered_vigor['pre'][nVDB]['locomotive'],stat='probability', color='gray',alpha = .3,label = 'Pre')
sns.histplot(clustered_vigor['CNO'][nVDB]['locomotive'],stat='probability', color='blue',alpha = .5,label = 'CNO')
sns.histplot(clustered_vigor['post'][nVDB]['locomotive'],stat='probability', color='gray',alpha = .7,label = 'Post')
plt.xlabel('Vigor (a.u)')
plt.title('Locomotive')
plt.legend()

plt.suptitle('Vigor of stationary and locomotive actions around CNO day\n Behaviors : Rearing , other , BTC , jump')
plt.savefig(output_folder + '/stat_loco_vigor_histograms_VDB.png', dpi=300)
plt.close()