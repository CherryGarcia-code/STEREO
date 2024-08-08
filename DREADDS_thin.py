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
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CMT_May24.pkl', 'rb')
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
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163','#969595', '#b4a7d6']
SSD_color = '#CC6677'
velocity_color = '#44AA99'
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
RNN_offset = 7
locomotive = 0
stationary = 1
cutoff = 15*minute
mice_to_remove = {'drd1_hm4di':['c528m10','c548m1'],'drd1_hm3dq':[],'controls':[],'a2a_hm4di':[],'a2a_hm3dq':[]}
for m in mice_to_remove[cohort]:
    del sandwich_days[m] # Bad infection
num_of_mice = len(list(sandwich_days.keys()))
EGO_LICKING = 0
ALLO_LICKING = 1
NO_LICKING = 2
lut = np.array([EGO_LICKING,EGO_LICKING,ALLO_LICKING,ALLO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
#%%Representative ethograms
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Ethograms/'
for m in sandwich_days:
    fig , ax = plt.subplots(nrows=3,sharex='all',figsize=(18,6))
    pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
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
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Summary'
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days))
clustered_fraction_of_frames = np.zeros((num_of_mice , 3 , num_of_days))
m_idx = 0
for m in sandwich_days:
    pre = np.copy(sandwich_days[m]['pre']['merged'])[:cutoff]
    CNO = np.copy(sandwich_days[m]['CNO']['merged'])[:cutoff]
    post = np.copy(sandwich_days[m]['post']['merged'])[:cutoff]
    for b in range(num_of_behaviors):
        fraction_of_frames[m_idx , b , PRE] = np.count_nonzero(pre==b)/np.count_nonzero(pre!=5)
        fraction_of_frames[m_idx, b, DURING] = np.count_nonzero(CNO == b) / np.count_nonzero(CNO!=5)
        fraction_of_frames[m_idx, b, POST] = np.count_nonzero(post == b) / np.count_nonzero(post!=5)
    pre = pre[pre!=5] # excluding BTC frames from analysis
    CNO = CNO[CNO != 5] # excluding BTC frames from analysis
    post = post[post != 5] # excluding BTC frames from analysis
    clustered_pre = lut[pre]
    clustered_CNO = lut[CNO]
    clustered_post = lut[post]
    for category in [EGO_LICKING,ALLO_LICKING,NO_LICKING]:
        clustered_fraction_of_frames[m_idx,category,PRE] = np.count_nonzero(clustered_pre==category)/np.count_nonzero(pre!=5)
        clustered_fraction_of_frames[m_idx, category, DURING] = np.count_nonzero(clustered_CNO == category) / np.count_nonzero(CNO!=5)
        clustered_fraction_of_frames[m_idx, category, POST] = np.count_nonzero(clustered_post == category) / np.count_nonzero(post!=5)
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
    plt.title('Change in behavior frequency around CNO day (first '+str(int(cutoff//minute))+'minutes)\n Behaviors: '+behaviors[b])
    plt.savefig(output_folder + '/V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

category2name = ['Egocentric licking','Allocentric licking','No licking']
category2colors = ['#5FFBF1','#D16BA5','#86A8E7']
for category in [EGO_LICKING,ALLO_LICKING,NO_LICKING]:
    plt.figure()
    mean = np.mean(clustered_fraction_of_frames[:,category,:],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,category,:], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,category,:].T, alpha=0.3, color='gray', marker='o')
    plt.bar(1, height=1, color='#009FE3', alpha=.3,label = 'CNO-day')
    plt.plot(mean, color=category2colors[category])
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=category2colors[category],capsize=2,capthick=1)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylabel('Fraction of frames')
    plt.yticks([0,0.5,1],[0,0.5,1])
    plt.ylim([0,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('CNO effect on prevalence of behavior: ' +category2name[category])
    plt.savefig(output_folder + '/V_graph_'+category2name[category].replace(' ','_')+'.png', dpi=300)
    plt.close()
#%%States & actions diagram for each day
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/States & Actions/'
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

#%% CDFs pf bout length
import matplotlib as mpl
cutoff=15*minute
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Bouts/'
labels_dict = {PRE:'Pre CNO',DURING:'CNO',POST:'Post CNO'}
colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,.5,]
longest_bout = 400
x_axis = np.arange(longest_bout)
cdf = np.empty((num_of_behaviors,3,num_of_mice,longest_bout))
cdf[:,:,:,:]=np.nan

for b in range(num_of_behaviors):
    plt.figure()
    m_idx=0
    for m in sandwich_days:
        preCNO = np.copy(sandwich_days[m]['pre']['merged'][:cutoff])
        CNO = np.copy(sandwich_days[m]['CNO']['merged'][:cutoff])
        postCNO = np.copy(sandwich_days[m]['post']['merged'][:cutoff])
        tr_idx=0
        for predictions in [preCNO,CNO,postCNO]:
            bouts_data = np.array(hf.segment_bouts(predictions,b,8)['length'])
            if bouts_data.size>0:
                for bl in x_axis:
                    cdf[b,tr_idx,m_idx,bl] = np.count_nonzero(bouts_data<=bl)/bouts_data.size
            tr_idx+=1
        m_idx+=1


    for tr_idx in range(3):
        mean = np.nanmean(cdf[b,tr_idx,:,:],axis=0)
        stderr = np.sqrt(np.nanvar(cdf[b,tr_idx,:,:],axis=0)/m_idx)
        plt.plot(x_axis,mean,color = colors_prime[tr_idx],alpha=.7,label = labels_dict[tr_idx])
        plt.fill_between(x_axis,y1=mean-stderr,y2=mean+stderr, color=colors_prime[tr_idx], alpha=.3)

    plt.xlabel('Bout length (# of frames)')
    plt.ylabel('${P}(|bout| \leq {x}$)')
    plt.title('CDF of bout length for behavior: '+behaviors[b])
    plt.legend()
    plt.savefig(output_folder+'bout_length_cdf'+behaviors[b]+'.png',dpi=300)
    plt.close()

#%% SSD statistics for each behavior
import matplotlib as mpl
cutoff=15*minute
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Movement_parameters/'
labels_dict = {PRE:'Pre CNO',DURING:'CNO',POST:'Post CNO'}
colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,.5,]
longest_bout = 400
x_axis = np.arange(0,150,0.05)
SSD = {'pre':{0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]} , 'CNO':{0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]},'post':{0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}}
for b in range(num_of_behaviors):
    for m in sandwich_days:
        preCNO = np.copy(sandwich_days[m]['pre']['merged'][:cutoff])
        pre_SSD = np.max(np.vstack([sandwich_days[m]['pre']['SSD']['cam1'][:cutoff],sandwich_days[m]['pre']['SSD']['cam2'][:cutoff]]),axis=0)
        CNO = np.copy(sandwich_days[m]['CNO']['merged'][:cutoff])
        CNO_SSD = np.max(np.vstack([sandwich_days[m]['CNO']['SSD']['cam1'][:cutoff],sandwich_days[m]['CNO']['SSD']['cam2'][:cutoff]]),axis=0)
        postCNO = np.copy(sandwich_days[m]['post']['merged'][:cutoff])
        post_SSD = np.max(np.vstack([sandwich_days[m]['post']['SSD']['cam1'][:cutoff],sandwich_days[m]['post']['SSD']['cam2'][:cutoff]]),axis=0)
        SSD['pre'][b].extend(pre_SSD[preCNO==b])
        SSD['CNO'][b].extend(CNO_SSD[CNO == b])
        SSD['post'][b].extend(post_SSD[postCNO == b])

for b in range(num_of_behaviors):
    fig,ax = plt.subplots(ncols=2)
    plt.sca(ax[0])
    sns.histplot(SSD['pre'][b],color = '#86A8E7',alpha = .7,stat = 'probability',label = 'Pre CNO',bins=100)
    sns.histplot(SSD['CNO'][b], color='#D16BA5', alpha=.7, stat='probability', label='During CNO', bins=100)
    sns.histplot(SSD['post'][b], color='#5FFBF1', alpha=.7, stat='probability', label='Post CNO', bins=100)
    plt.legend()
    plt.xlabel('SSD(a.u)')
    plt.title('Histogram')
    plt.legend()
    plt.sca(ax[1])
    pre_cdf = []
    CNO_cdf = []
    post_cdf = []
    for x in x_axis:
        if len(SSD['pre'][b])>0:
            pre_cdf.append(np.count_nonzero(SSD['pre'][b]<=x)/len(SSD['pre'][b]))
        if len(SSD['CNO'][b])>0:
            CNO_cdf.append(np.count_nonzero(SSD['CNO'][b] <= x) / len(SSD['CNO'][b]))
        if len(SSD['post'][b])>0:
            post_cdf.append(np.count_nonzero(SSD['post'][b] <= x) / len(SSD['post'][b]))
    if len(pre_cdf)>0: plt.plot(x_axis,pre_cdf , color ='#86A8E7',alpha=.7,label = 'Pre CNO')
    if len(CNO_cdf)>0:plt.plot(x_axis, CNO_cdf, color='#D16BA5', alpha=.7, label='During CNO')
    if len(post_cdf)>0:plt.plot(x_axis, post_cdf, color='#5FFBF1', alpha=.7, label='Post CNO')
    plt.xlabel('SSD(a.u)')
    plt.ylabel('${P(SSD<=x)}$')
    plt.title('CDF')
    plt.legend()
    plt.suptitle('CNO effect on SSD statistics for behavior:'+behaviors[b])
    fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
    plt.savefig(output_folder+'/SSD_stat_'+behaviors[b]+'.png',dpi=300)
    plt.close()

#%% CNO effect on velocity during floor licking and locomotion
FL = 3
LOC = 7
import matplotlib as mpl
cutoff=15*minute
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Movement_parameters/'
labels_dict = {PRE:'Pre CNO',DURING:'CNO',POST:'Post CNO'}
colors_prime = ['gray','blue','gray']
alpha_prime = [.3,.5,.7,.5,]
longest_bout = 400
x_axis = [np.arange(0,0.1,0.001),np.arange(0,0.25,0.001)]

behaviors_of_interest = [FL , LOC]
vel = {'pre':{FL:[] , LOC:[]} , 'CNO':{FL:[] , LOC:[]},'post':{FL:[] , LOC:[]}}
for b in behaviors_of_interest:
    for m in sandwich_days:
        preCNO = np.copy(sandwich_days[m]['pre']['merged'][:cutoff])
        pre_vel = np.copy(sandwich_days[m]['pre']['topcam']['velocity'][:cutoff])*FPS
        CNO = np.copy(sandwich_days[m]['CNO']['merged'][:cutoff])
        CNO_vel = np.copy(sandwich_days[m]['CNO']['topcam']['velocity'][:cutoff])*FPS
        postCNO = np.copy(sandwich_days[m]['post']['merged'][:cutoff])
        post_vel = np.copy(sandwich_days[m]['post']['topcam']['velocity'][:cutoff])*FPS
        vel['pre'][b].extend(hf.remove_IQR_outliers(pre_vel[preCNO==b]))
        vel['CNO'][b].extend(hf.remove_IQR_outliers(CNO_vel[CNO == b]))
        vel['post'][b].extend(hf.remove_IQR_outliers(post_vel[postCNO == b]))
axis_idx=0
for b in behaviors_of_interest:
    fig,ax = plt.subplots(ncols=2)
    plt.sca(ax[0])
    sns.histplot(vel['pre'][b],color = '#86A8E7',alpha = .7,stat = 'probability',label = 'Pre CNO',bins=100)
    sns.histplot(vel['CNO'][b], color='#D16BA5', alpha=.7, stat='probability', label='During CNO', bins=100)
    sns.histplot(vel['post'][b], color='#5FFBF1', alpha=.7, stat='probability', label='Post CNO', bins=100)
    plt.legend()
    plt.xlabel('SSD(a.u)')
    plt.title('Histogram')
    plt.legend()
    plt.sca(ax[1])
    pre_cdf = []
    CNO_cdf = []
    post_cdf = []
    for x in x_axis[axis_idx]:
        if len(vel['pre'][b])>0:
            pre_cdf.append(np.count_nonzero(vel['pre'][b]<=x)/len(vel['pre'][b]))
        if len(vel['CNO'][b])>0:
            CNO_cdf.append(np.count_nonzero(vel['CNO'][b] <= x) / len(vel['CNO'][b]))
        if len(vel['post'][b])>0:
            post_cdf.append(np.count_nonzero(vel['post'][b] <= x) / len(vel['post'][b]))
    if len(pre_cdf)>0: plt.plot(x_axis[axis_idx],pre_cdf , color ='#86A8E7',alpha=.7,label = 'Pre CNO')
    if len(CNO_cdf)>0:plt.plot(x_axis[axis_idx], CNO_cdf, color='#D16BA5', alpha=.7, label='During CNO')
    if len(post_cdf)>0:plt.plot(x_axis[axis_idx], post_cdf, color='#5FFBF1', alpha=.7, label='Post CNO')
    plt.xlabel('velocity(m/s)')
    plt.ylabel('${P(vel<=x)}$')
    plt.title('CDF')
    plt.legend()
    plt.suptitle('CNO effect on velocity statistics for behavior:'+behaviors[b])
    fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
    plt.savefig(output_folder+'/vel_stat_'+behaviors[b]+'.png',dpi=300)
    plt.close()
    axis_idx+=1







