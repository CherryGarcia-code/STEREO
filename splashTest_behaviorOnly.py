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
import networkx as nx
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CTM_May24.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT_May24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Stationary','Locomotion',  'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163','#969595', '#b4a7d6']
SSD_color = '#CC6677'
velocity_color = '#44AA99'
FPS=15
sample_rate=15
second = 15
minute = 60*second
mpl.rcParams['xtick.labelsize']=10
mpl.rcParams['ytick.labelsize']=10
mpl.rcParams['legend.fontsize']=8
VDB = 0
nVDB = 1

trials  = ['saline1','saline2','saline3','splashTest','BaselineDay1','BaselineDay2','BaselineDay3']
cohorts = ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq','a2a_opto']
num_of_trials = len(trials)
num_of_behaviors = len(behaviors)
splashTest_days = {}
counter=0
for c in cohorts:
    for m in CMT[c]:
        if 'splashTest' in CMT[c][m]:
            splashTest_days[m] = {}
            print(c,m,CMT[c][m].keys())
            if c =='a2a_opto':
                splashTest_days[m]['pre1'] = CMT[c][m]['BaselineDay1']
                if m == 'cA180m6':
                    splashTest_days[m]['pre2'] = CMT[c][m]['BaselineDay2A2ACreChR2']
                else:
                    splashTest_days[m]['pre2'] = CMT[c][m]['BaselineDay2']
                splashTest_days[m]['pre3'] = CMT[c][m]['BaselineDay3']
                splashTest_days[m]['splashTest'] = CMT[c][m]['splashTest']
            else:
                try:
                    splashTest_days[m]['pre1'] = CMT[c][m]['saline1']
                    splashTest_days[m]['pre2'] = CMT[c][m]['saline2']
                    splashTest_days[m]['pre3'] = CMT[c][m]['saline3']
                    splashTest_days[m]['splashTest'] = CMT[c][m]['splashTest']
                except KeyError:
                    counter+=1
                    del splashTest_days[m]

print('Mice with missing days : ',counter)
del CMT
del CTM

#%% Ethograms
RNN_offset = 7
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Ethograms/'
for m in splashTest_days:
    fig, ax = plt.subplots(nrows=4, sharex='all', figsize=(18, 6))
    print(m , splashTest_days[m].keys())
    pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
    pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
    pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
    st = np.copy(splashTest_days[m]['splashTest']['merged'])
    pre1 = np.reshape(pre1,(1,pre1.shape[0]))
    pre2 = np.reshape(pre2, (1, pre2.shape[0]))
    pre3 = np.reshape(pre3, (1, pre3.shape[0]))
    st = np.reshape(st, (1, st.shape[0]))
    cbar_ax = fig.add_axes([.88, .3, .01, .4])
    ax[0] = sns.heatmap(pre1, yticklabels=[''], cmap=colors,
                        cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5, 'drawedges': True},
                        cbar_ax=cbar_ax, vmin=0, vmax=num_of_behaviors - 1, ax=ax[0])
    plt.sca(ax[0])
    plt.ylabel('Baseline1',va='center')
    cbar = ax[0].collections[0].colorbar
    cbar.set_ticklabels(behaviors)

    ax[1] = sns.heatmap(pre2, yticklabels=[''], cmap=colors,cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[1])
    plt.sca(ax[1])
    plt.ylabel('Baseline2', va='center')
    ax[2] = sns.heatmap(pre3, yticklabels=[''], cmap=colors,cbar = False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[2])
    plt.sca(ax[2])
    plt.ylabel('Baseline3', va='center')

    ax[3] = sns.heatmap(st, yticklabels=[''], cmap=colors, cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[3])
    plt.sca(ax[3])
    plt.ylabel('Splash-test', va='center')
    T = np.max([pre1.size,pre2.size,pre3.size,st.size])
    plt.sca(ax[3])
    plt.xticks(np.arange(0,T,900),(np.arange(RNN_offset,T+RNN_offset,900)/900).astype(int),rotation=0)
    plt.xlabel('Time (m)')
    plt.suptitle(m)
    fig.tight_layout(rect=[0.02, 0.01, .86, 0.95], pad=1)
    plt.savefig(output_folder+m+'.png',dpi=300)
    plt.close()

#%% Bar garphs
PRE1 = 0
PRE2 = 1
PRE3 = 2
ST = 3
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Summary/'
num_of_mice = len(splashTest_days.keys())
fraction_of_frames = np.zeros((num_of_behaviors,4,num_of_mice))
labels_dict = {PRE1:'Baseline1',PRE2:'Baseline2',PRE3:'Baseline3',ST:'Splash-test'}
m_idx=0
for m in splashTest_days:
    pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
    pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
    pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
    st = np.copy(splashTest_days[m]['splashTest']['merged'])
    for b in range(len(behaviors)):
        fraction_of_frames[b, PRE1, m_idx] = np.count_nonzero(pre1 == b) / pre1.size
        fraction_of_frames[b, PRE2, m_idx] = np.count_nonzero(pre2 == b) / pre2.size
        fraction_of_frames[b, PRE3, m_idx] = np.count_nonzero(pre3 == b) / pre3.size
        fraction_of_frames[b, ST, m_idx] = np.count_nonzero(st == b) / st.size
    m_idx+=1
plt.figure(figsize=(12,8))
for t_idx in [PRE1,PRE2,PRE3,ST]:

    mean = np.nanmean(fraction_of_frames[:,t_idx,:],axis=1)
    stderr = np.sqrt(np.nanvar(fraction_of_frames[:,t_idx,:],axis=1)/num_of_mice)
    plt.bar(np.arange(t_idx,num_of_behaviors*4,4),height = mean , yerr = stderr,capsize=2,alpha = (t_idx+1)/4,label = labels_dict[t_idx],color=colors)
    plt.vlines(x=np.arange(3.5,num_of_behaviors*4,4),ymin=0,ymax=1,ls='--',lw=1,colors='gray')

plt.legend()
plt.xticks(np.arange(1.33,num_of_behaviors*4,4),behaviors)
plt.xlabel('Day')
plt.suptitle('Baseline days compared to splash test day')
plt.savefig(output_folder+'splashTest_behavior_distribution.png',dpi=300)
plt.close()
#%% States & actions diagram
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/States&Actions/'
num_of_mice = len(splashTest_days.keys())
transition_prob = np.zeros((4,num_of_mice,num_of_behaviors, num_of_behaviors))
time_spent=np.zeros((4,num_of_mice,num_of_behaviors))
mice_mean_transition_prob = np.zeros((4,num_of_behaviors, num_of_behaviors))
mice_mean_time_spent = np.zeros((4,num_of_behaviors))
m_idx=0
for m in splashTest_days:
    pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
    pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
    pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
    st = np.copy(splashTest_days[m]['splashTest']['merged'])
    t_idx=0
    for predictions in [pre1,pre2,pre3,st]:
        for i in range(predictions.size-1):
            transition_prob[t_idx,m_idx,predictions[i],predictions[i+1]]+=1
            time_spent[t_idx,m_idx,predictions[i]]+=1
        time_spent[t_idx, m_idx, predictions[-1]] += 1
        time_spent[t_idx, m_idx, :] /= np.sum(time_spent[t_idx, m_idx, :])
        np.fill_diagonal(transition_prob[t_idx, m_idx, :, :], 0)
        for b in range(num_of_behaviors):
            transition_prob[t_idx,m_idx,b,:]/=np.sum(transition_prob[t_idx,m_idx,b,:])
        mice_mean_transition_prob[t_idx,:,:] = np.sum(transition_prob[t_idx,:,:,:],axis=0) / float(num_of_mice)
        mice_mean_time_spent[t_idx,:] = np.sum(time_spent[t_idx,:,:], axis=0) / float(num_of_mice)
        t_idx += 1
    m_idx += 1
print('Done collecting data')
mice_mean_time_spent*=1000
transition_threshold = 1.0/(num_of_behaviors**2) # randomly selecting behavior and randomly selecting to move to the nexet behavior
aboveChance_alpha=0.8
belowChance_alpha=0.12
baseline_transiton_prob = np.nanmean(mice_mean_transition_prob[:3,:,:],axis=0)
baseline_time_spent = np.nanmean(mice_mean_time_spent[:3, :], axis=0)
plt.figure()
edges = []
edge_colors = []
edge_width = []
for b_cur in range(num_of_behaviors):
    for b_next in range(num_of_behaviors):
        edges.append((b_cur, b_next, baseline_transiton_prob[b_cur, b_next] * 5))
        if baseline_transiton_prob[b_cur, b_next] >= transition_threshold:
            edge_colors.append([0.294, 0, 0.51, aboveChance_alpha])
        else:
            edge_colors.append([0.294, 0, 0.51, belowChance_alpha])
G = nx.DiGraph()
G.add_weighted_edges_from(edges, color='red')
weights = nx.get_edge_attributes(G, 'weight').values()
plt.title('Mean Baseline days\n$\pi$[b$_t$,b$_{t+1}$]')
nx.draw(G, pos=nx.circular_layout(G), edge_color=edge_colors, node_color=colors, width=list(weights), node_size=baseline_time_spent,
        arrowstyle='->'
        , arrowsize=15, connectionstyle='arc3, rad = 0.1', with_labels=False)
plt.savefig(output_folder + 'states_actions_mean_baseline_days.png', dpi=300)
plt.close()


diff_transition_prob = mice_mean_transition_prob[3,:,:]-baseline_transiton_prob

plt.figure()
edges = []
edge_colors = []
edge_width = []
for b_cur in range(num_of_behaviors):
    for b_next in range(num_of_behaviors):
        edges.append((b_cur, b_next, diff_transition_prob[ b_cur, b_next] * 20))
        if diff_transition_prob[ b_cur, b_next] >= transition_threshold :
            color = [0.29, 0, 0.75,aboveChance_alpha]
        elif diff_transition_prob[ b_cur, b_next] <= -transition_threshold:
            color = [1, 0,0,aboveChance_alpha]
        elif diff_transition_prob[b_cur, b_next] <= transition_threshold and diff_transition_prob[ b_cur, b_next] >0:
            color = [0.29, 0, 0.75,belowChance_alpha]
        else:
            color =[1, 0,0 ,belowChance_alpha]
        edge_colors.append(color)
G = nx.DiGraph()
G.add_weighted_edges_from(edges, color='red')
weights = nx.get_edge_attributes(G, 'weight').values()
plt.title('Splash Test \n$\Delta$ $\pi$[b$_t$,b$_{t+1}$]' )
nx.draw(G, pos=nx.circular_layout(G), edge_color=edge_colors, node_color=colors, width=list(weights), node_size=mice_mean_time_spent[3],
        arrowstyle='->', arrowsize=15, connectionstyle='arc3, rad = 0.1', with_labels=False)
plt.savefig(output_folder + 'states_actions_diff_SplashTest.png', dpi=300)
plt.close()

#%% CDFs pf bout length
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Bouts/'
labels_dict = {PRE1:'Baseline1',PRE2:'Baseline2',PRE3:'Baseline3',ST:'Splash-test'}
colors_prime = ['gray','gray','gray','blue']
alpha_prime = [.3,.5,.7,.5,]
longest_bout = 400
x_axis = np.arange(longest_bout)
num_of_mice = len(splashTest_days.keys())
cdf = np.empty((num_of_behaviors,4,num_of_mice,longest_bout))
cdf[:,:,:,:]=np.nan

for b in range(num_of_behaviors):
    plt.figure()
    m_idx=0
    for m in splashTest_days:
        pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
        pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
        pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
        st = np.copy(splashTest_days[m]['splashTest']['merged'])
        tr_idx=0
        for predictions in [pre1,pre2,pre3,st]:
            bouts_data = np.array(hf.segment_bouts(predictions,b,8)['length'])
            if bouts_data.size>0:
                for bl in x_axis:
                    cdf[b,tr_idx,m_idx,bl] = np.count_nonzero(bouts_data<=bl)/bouts_data.size
            tr_idx+=1
        m_idx+=1


    for tr_idx in range(4):
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

#%% Histograms of SSD for all behaviors in baseline VS splash test
PRE1 = 0
PRE2 = 1
PRE3 = 2
ST = 3
x_axis = np.arange(0,200,0.1)
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Movement_parameters/'
num_of_mice = len(splashTest_days.keys())
baseline_ssd_tot = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
st_ssd_tot = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for m in splashTest_days:
    pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
    pre1_SSD = np.max(np.vstack([splashTest_days[m]['pre1']['SSD']['cam1'], splashTest_days[m]['pre1']['SSD']['cam2']]), axis=0)
    pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
    pre2_SSD = np.max(np.vstack([splashTest_days[m]['pre2']['SSD']['cam1'], splashTest_days[m]['pre2']['SSD']['cam2']]), axis = 0)
    pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
    pre3_SSD = np.max(np.vstack([splashTest_days[m]['pre3']['SSD']['cam1'], splashTest_days[m]['pre3']['SSD']['cam2']]), axis = 0)
    st = np.copy(splashTest_days[m]['splashTest']['merged'])
    st_SSD = np.max(np.vstack([splashTest_days[m]['splashTest']['SSD']['cam1'], splashTest_days[m]['splashTest']['SSD']['cam2']]), axis = 0)
    for b in range(len(behaviors)):
        baseline_ssd_tot[b].extend(pre1_SSD[pre1==b])
        baseline_ssd_tot[b].extend(pre2_SSD[pre2 == b])
        baseline_ssd_tot[b].extend(pre3_SSD[pre3 == b])
        st_ssd_tot[b].extend(st_SSD[st==b])

for b in range(num_of_behaviors):
    fig,ax = plt.subplots(ncols=2)
    plt.sca(ax[0])
    sns.histplot(baseline_ssd_tot[b],color = 'gray',alpha = .3,stat = 'probability',label = 'Baseline',bins=100)
    sns.histplot(st_ssd_tot[b], color='blue', alpha=.3, stat = 'probability',label='Splash Test',bins=100)
    plt.title('Histogram')
    plt.xlabel('SSD (a.u)')
    plt.legend()
    plt.sca(ax[1])
    baseline_cdf = []
    st_cdf = []
    for x in x_axis:
        baseline_cdf.append(np.count_nonzero(baseline_ssd_tot[b]<=x)/len(baseline_ssd_tot[b]))
        st_cdf.append(np.count_nonzero(st_ssd_tot[b] <= x) / len(st_ssd_tot[b]))
    plt.plot(x_axis , baseline_cdf, c = 'gray',alpha=.7,label = 'Baseline')
    plt.plot(x_axis,st_cdf,c = 'blue',alpha=.7,label = 'Splash Test')
    plt.title('CDF')
    plt.ylabel('${P}(|SSD| \leq {x}$)')
    plt.legend()
    plt.suptitle('SSD statistics for behavior : '+behaviors[b])
    fig.tight_layout(rect=[0.05, 0.05, .9, 0.9], pad=.1)
    plt.savefig(output_folder+'SSD_'+behaviors[b]+'.png',dpi=300)
    plt.close()

#%% CDF of SSD for all behaviors in baseline VS splash test
"""Comment: At the moment, averaging over mice """
PRE1 = 0
PRE2 = 1
PRE3 = 2
ST = 3
x_axis = np.arange(0,200,0.1)
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Movement_parameters/'
num_of_mice = len(splashTest_days.keys())
m_idx=0
baseline_ssd_tot = np.empty((num_of_mice ,3,num_of_behaviors, x_axis.size))
baseline_ssd_tot[:,:,:,:]=np.nan
st_ssd_tot = np.empty((num_of_mice ,num_of_behaviors, x_axis.size))
st_ssd_tot[:,:,:]=np.nan
colors_prime = ['gray','gray','gray','blue']
alpha_prime = [.3,.5,.7,.5,]
for m in splashTest_days:
    pre1 = np.copy(splashTest_days[m]['pre1']['merged'])
    pre1_SSD = np.max(np.vstack([splashTest_days[m]['pre1']['SSD']['cam1'], splashTest_days[m]['pre1']['SSD']['cam2']]), axis=0)
    pre2 = np.copy(splashTest_days[m]['pre2']['merged'])
    pre2_SSD = np.max(np.vstack([splashTest_days[m]['pre2']['SSD']['cam1'], splashTest_days[m]['pre2']['SSD']['cam2']]), axis = 0)
    pre3 = np.copy(splashTest_days[m]['pre3']['merged'])
    pre3_SSD = np.max(np.vstack([splashTest_days[m]['pre3']['SSD']['cam1'], splashTest_days[m]['pre3']['SSD']['cam2']]), axis = 0)
    st = np.copy(splashTest_days[m]['splashTest']['merged'])
    st_SSD = np.max(np.vstack([splashTest_days[m]['splashTest']['SSD']['cam1'], splashTest_days[m]['splashTest']['SSD']['cam2']]), axis = 0)
    for b in range(len(behaviors)):
        x_idx=0
        pre1_SSD_b = pre1_SSD[pre1 == b]
        pre2_SSD_b = pre2_SSD[pre2 == b]
        pre3_SSD_b = pre3_SSD[pre3 == b]
        st_SSD_b = st_SSD[st == b]
        for x in x_axis:
            if pre1_SSD_b.size>0:
                baseline_ssd_tot[m_idx,PRE1,b,x_idx] = np.count_nonzero(pre1_SSD_b<=x)/pre1_SSD_b.size
            if pre2_SSD_b.size>0:
                baseline_ssd_tot[m_idx, PRE2, b, x_idx] = np.count_nonzero(pre2_SSD_b <= x) / pre2_SSD_b.size
            if pre3_SSD_b.size>0:
                baseline_ssd_tot[m_idx, PRE3, b, x_idx] = np.count_nonzero(pre3_SSD_b <= x) / pre3_SSD_b.size
            if st_SSD_b.size>0:
                st_ssd_tot[m_idx,b,x_idx] = np.count_nonzero(st_SSD_b<=x)/st_SSD_b.size
            x_idx+=1
    m_idx+=1

for b in range(num_of_behaviors):
    cdf_pre1_mean = np.nanmean(baseline_ssd_tot[:,PRE1,b],axis=0)
    cdf_pre1_stderr = np.sqrt(np.nanvar(baseline_ssd_tot[:, PRE1, b], axis=0)/num_of_mice)
    cdf_pre2_mean = np.nanmean(baseline_ssd_tot[:, PRE2, b], axis=0)
    cdf_pre2_stderr = np.sqrt(np.nanvar(baseline_ssd_tot[:, PRE2, b], axis=0) / num_of_mice)
    cdf_pre3_mean = np.nanmean(baseline_ssd_tot[:, PRE3, b], axis=0)
    cdf_pre3_stderr = np.sqrt(np.nanvar(baseline_ssd_tot[:, PRE3, b], axis=0) / num_of_mice)
    cdf_st_mean = np.nanmean(st_ssd_tot[:,b], axis=0)
    cdf_st_stderr = np.sqrt(np.nanvar(st_ssd_tot[:, b], axis=0) / num_of_mice)
    plt.figure()
    plt.plot(x_axis,cdf_pre1_mean,c = 'gray' ,alpha = .4 ,label = 'Baseline D1')
    plt.fill_between(x_axis,y1=cdf_pre1_mean-cdf_pre1_stderr,y2=cdf_pre1_mean+cdf_pre1_stderr,color = 'gray' ,alpha=.3)
    plt.plot(x_axis, cdf_pre2_mean, c='gray', alpha=.6, label='Baseline D2')
    plt.fill_between(x_axis, y1=cdf_pre2_mean - cdf_pre2_stderr, y2=cdf_pre2_mean + cdf_pre2_stderr, color='gray', alpha=.5)
    plt.plot(x_axis, cdf_pre3_mean, c='gray', alpha=.8, label='Baseline D3')
    plt.fill_between(x_axis, y1=cdf_pre3_mean - cdf_pre3_stderr, y2=cdf_pre3_mean + cdf_pre3_stderr, color='gray', alpha=.7)
    plt.plot(x_axis, cdf_st_mean, c='blue', alpha=.8, label='Splash Test')
    plt.fill_between(x_axis, y1=cdf_st_mean - cdf_st_stderr, y2=cdf_st_mean + cdf_st_stderr, color='blue', alpha=.6)
    plt.title('CDF of SSD for behavior : '+behaviors[b])
    plt.xlabel('SSD (a.u)')
    plt.ylabel('${P}(|SSD| \leq {x}$)')
    plt.legend()
    plt.savefig(output_folder+'CDF_SSD_'+behaviors[b]+'.png',dpi=300)
    plt.close()