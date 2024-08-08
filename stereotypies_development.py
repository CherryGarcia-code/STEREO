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
GR = 0
BL = 1
WL = 2
FL = 3
trials  = ['saline1','saline2','saline3','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']
cohorts = ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq','a2a_opto']
num_of_trials = len(trials)
num_of_behaviors = len(behaviors)
MT = {}
TM = {}
for c in cohorts:
    for m in CMT[c]:
        MT[m] = CMT[c][m]
for cohort in cohorts:
    for mouse in CMT[cohort].keys():
        for t in CMT[cohort][mouse].keys():
            if t not in TM.keys():
                TM[t] = {}
            TM[t][mouse] = CMT[cohort][mouse][t]
del CMT
del CTM

EGO_LICKING = 0
ALLO_LICKING = 1
NO_LICKING = 2
lut = np.array([EGO_LICKING,EGO_LICKING,ALLO_LICKING,ALLO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
#%% Only means and stderr of behaviors across days
output_folder = root_folder+'/Figures/Stereotypies_development/Summary/'
num_of_mice = []
max_num_of_mice = 0
for t in trials:
    max_num_of_mice = np.max([max_num_of_mice,len(TM[t].keys())])
    num_of_mice.append(len(TM[t].keys()))
fraction_of_frames = np.empty((num_of_behaviors,num_of_trials,max_num_of_mice))
fraction_of_frames[:,:,:] = np.nan
clustered_fraction_of_frames = np.empty((3,num_of_trials,max_num_of_mice))
clustered_fraction_of_frames[:,:,:] = np.nan
t_idx = 0
for t in trials:
    m_idx = 0
    for m in TM[t]:
        print(t,m)
        predictions = TM[t][m]['merged']
        for b in range(len(behaviors)):
            fraction_of_frames[b,t_idx,m_idx]= np.count_nonzero(predictions==b)/predictions.size
        clustered_predictions = lut[predictions]
        clustered_fraction_of_frames[EGO_LICKING, t_idx, m_idx] = np.count_nonzero(clustered_predictions == EGO_LICKING) / predictions.size
        clustered_fraction_of_frames[ALLO_LICKING, t_idx, m_idx] = np.count_nonzero(clustered_predictions == ALLO_LICKING) / predictions.size
        clustered_fraction_of_frames[NO_LICKING, t_idx, m_idx] = np.count_nonzero(clustered_predictions == NO_LICKING) / predictions.size
        m_idx+=1
    t_idx+=1
plt.figure()
for b in range(num_of_behaviors):
    mean = np.nanmean(fraction_of_frames[b],axis=1)
    stderr = np.sqrt(np.nanvar(fraction_of_frames[b],axis=1)/np.array(num_of_mice))
    plt.plot(mean,color = colors[b], label = behaviors[b],ls='--',lw=1,marker='o',ms=2,markerfacecolor='white')
    plt.errorbar(np.arange(num_of_trials),y=mean,yerr=stderr,color=colors[b],capsize=2,capthick=1,ls='',elinewidth=1)
plt.yticks(np.arange(0,1,0.1),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Fraction of frames')
plt.xticks(np.arange(8),trials)
plt.title('Fraction of frames')
plt.xlabel('Day')
plt.legend()
plt.savefig(output_folder+'fraction_of_frames_linePlots.png',dpi=300)
plt.close()

ego_licking_mean = np.nanmean(clustered_fraction_of_frames[EGO_LICKING],axis=1)
ego_licking_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[EGO_LICKING],axis=1)/np.array(num_of_mice))
allo_licking_mean = np.nanmean(clustered_fraction_of_frames[ALLO_LICKING],axis=1)
allo_licking_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[ALLO_LICKING],axis=1)/np.array(num_of_mice))
no_licking_mean = np.nanmean(clustered_fraction_of_frames[NO_LICKING],axis=1)
no_licking_stderr = np.sqrt(np.nanvar(clustered_fraction_of_frames[NO_LICKING],axis=1)/np.array(num_of_mice))

plt.plot(ego_licking_mean,color = '#5FFBF1', label = 'Egocentric licking',ls='--',lw=1,marker='o',markerfacecolor='white')
plt.errorbar(np.arange(num_of_trials),y=ego_licking_mean,yerr=ego_licking_stderr,color= '#5FFBF1',capsize=2,capthick=1,ls='')
plt.plot(allo_licking_mean,color = '#D16BA5', label = 'Allocentric licking',ls='--',lw=1,marker='o',markerfacecolor='white')
plt.errorbar(np.arange(num_of_trials),y=allo_licking_mean,yerr=allo_licking_stderr,color='#D16BA5',capsize=2,capthick=1,ls='')
plt.plot(no_licking_mean,color = '#86A8E7', label = 'No licking behaviors',ls='--',lw=1,marker='o',markerfacecolor='white')
plt.errorbar(np.arange(num_of_trials),y=no_licking_mean,yerr=no_licking_stderr,color='#86A8E7',capsize=2,capthick=1,ls='')
plt.yticks(np.arange(0,1,0.1),[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.ylabel('Fraction of frames')
plt.xticks(np.arange(8),trials)
plt.title('Fraction of frames')
plt.xlabel('Day')
plt.legend()
plt.savefig(output_folder+'fraction_of_frames_linePlots_clustered.png',dpi=300)
plt.close()


#%%Change in SSD for each behavior, across the days of stereotypies development
output_folder = root_folder+'/Figures/Stereotypies_development/Movement_parameters/'
x_axis = np.arange(0,200,0.5)
SSD = {}
behaviors_of_interset = [GR , BL , FL , WL]
for b in behaviors_of_interset:
    SSD[b] = {}
    for tr in trials:
        SSD[b][tr] = []
for tr in trials:
    for m in TM[tr]:
        predictions = TM[tr][m]['merged']
        session_SSD = np.max(np.vstack([TM[tr][m]['SSD']['cam1'],TM[tr][m]['SSD']['cam2']]), axis=0)
        for b in behaviors_of_interset:
            SSD[b][tr].extend(session_SSD[predictions==b])
print('Finshed collecting data')
colors_prime = ['gray','gray','gray','blue','turquoise','green','orange','red']
alpha_prime = [.3,.5,.7,.5,.5,.5,.5,.5]
for b in behaviors_of_interset:
    print('Plotting stat for ',behaviors[b])
    fig , ax = plt.subplots(ncols = 2,figsize=(14,6))
    plt.sca(ax[0])
    tr_idx = 0
    for tr in trials:
        print('Hist')
        sns.histplot(SSD[b][tr],label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx],bins=100)
        tr_idx += 1
    plt.xlabel('SSD (a.u)')
    plt.title('Historgam')
    plt.legend()
    plt.sca(ax[1])
    tr_idx = 0
    for tr in trials:
        print('CDF')
        if len(SSD[b][tr])>0:
            b_tr_cdf = []
            for x in x_axis:
                b_tr_cdf.append(np.count_nonzero(SSD[b][tr] <= x) / len(SSD[b][tr]))
            plt.plot(x_axis,b_tr_cdf,c = colors_prime[tr_idx],alpha = alpha_prime[tr_idx],label = tr)
        tr_idx+=1
    plt.xlabel('SSD (a.u)')
    plt.title('CDF')
    plt.legend()
    plt.suptitle('SSD stat across the days of the development of stereotypies\n Behavior : '+behaviors[b])
    plt.savefig(output_folder+'/SSD_stat_'+behaviors[b]+'.png',dpi=300)
    plt.close()

#%% Distance travelled mean and stderr across days
output_folder = root_folder+'/Figures/Stereotypies_development/Movement_parameters/'
num_of_mice = []
max_num_of_mice = 0
for t in trials:
    max_num_of_mice = np.max([max_num_of_mice,len(TM[t].keys())])
    num_of_mice.append(len(TM[t].keys()))
distance_travelled = np.zeros((max_num_of_mice,num_of_trials))
t_idx=0
for t in trials:
    m_idx = 0
    for m in TM[t]:
        distance_travelled[m_idx,t_idx] = np.sum(TM[t][m]['topcam']['velocity'])
        m_idx+=1
    t_idx+=1

mean = np.nanmean(distance_travelled,axis=0)
stderr = np.sqrt(np.nanvar(distance_travelled,axis=0)/np.array(num_of_mice))
plt.figure(figsize=(8,6))
plt.plot(np.arange(num_of_trials),mean,ls='--',marker='o',lw=1,markerfacecolor='white',c='k')
plt.errorbar(np.arange(num_of_trials),y=mean,yerr=stderr,color='k',capsize=2,capthick=1,ls='')
plt.title('Distance travelled for over the course of orofacial stereotypies development')
plt.xlabel('Day')
plt.xticks(np.arange(num_of_trials),trials)
plt.ylabel('Distance Travelled(m)')
plt.savefig(output_folder+'distance_travelled.pdf',dpi=300)
plt.close()

#%% Saline 1,2,3 to salineOnly comparison
output_folder = root_folder+'/Figures/Stereotypies_development/Summary/'
num_of_mice = []
for t in ['saline1','saline2','saline3','salineOnly']:
    num_of_mice.append(len(TM[t].keys()))
max_num_of_mice = np.max(num_of_mice)
print(num_of_mice)
fraction_of_frames = np.empty((num_of_behaviors,4,max_num_of_mice))
fraction_of_frames[:,:,:] = np.nan
t_idx = 0
plt.figure(figsize=(12,8))
for t in ['saline1','saline2','saline3','salineOnly']:
    m_idx = 0
    for m in TM[t]:
        print(t,m)
        predictions = TM[t][m]['merged']
        for b in range(len(behaviors)):
            fraction_of_frames[b, t_idx, m_idx] = np.count_nonzero(predictions == b) / predictions.size
        m_idx+=1
    mean = np.nanmean(fraction_of_frames[:,t_idx,:],axis=1)
    stderr = np.sqrt(np.nanvar(fraction_of_frames[:,t_idx,:],axis=1)/num_of_mice[t_idx])
    plt.bar(np.arange(t_idx,num_of_behaviors*4,4),height = mean , yerr = stderr,capsize=2,alpha = (t_idx+1)/4,label = t,color=colors)
    plt.vlines(x=np.arange(3.5,num_of_behaviors*4,4),ymin=0,ymax=1,ls='--',lw=1,colors='gray')
    t_idx += 1
plt.legend()
plt.xticks(np.arange(1.33,num_of_behaviors*4,4),behaviors)
plt.xlabel('Day')
plt.suptitle('Cocaine long lasting effect\n Pre cocaine saline days compared to post cocaine saline day')
plt.savefig(output_folder+'cocaine_long_term_effect.png',dpi=300)
plt.close()
#%% States and actions diagrams for transitions probabilities and time spent in each behavior
output_folder = root_folder+'/Figures/Stereotypies_development/States & Actions/'
trials_subset = np.copy(trials)
trials_subset = np.append(trials_subset,'splashTest')
num_of_subset_trials = trials_subset.size
num_of_mice = []
for t in trials_subset:
    num_of_mice.append(len(TM[t].keys()))
max_num_of_mice = np.max(num_of_mice)
transition_prob = np.zeros((num_of_subset_trials,max_num_of_mice,num_of_behaviors, num_of_behaviors))
time_spent=np.zeros((num_of_subset_trials,max_num_of_mice,num_of_behaviors))
mice_mean_transition_prob = np.zeros((num_of_subset_trials,num_of_behaviors, num_of_behaviors))
mice_mean_time_spent = np.zeros((num_of_subset_trials,num_of_behaviors))
t_idx=0
for t in trials_subset:
    m_idx = 0
    for m in TM[t]:
        print(t,m)
        predictions = TM[t][m]['merged']
        for i in range(predictions.size-1):
            transition_prob[t_idx,m_idx,predictions[i],predictions[i+1]]+=1
            time_spent[t_idx,m_idx,predictions[i]]+=1
        time_spent[t_idx, m_idx, predictions[-1]] += 1
        time_spent[t_idx, m_idx, :] /= np.sum(time_spent[t_idx, m_idx, :])
        np.fill_diagonal(transition_prob[t_idx, m_idx, :, :], 0)
        for b in range(num_of_behaviors):
            transition_prob[t_idx,m_idx,b,:]/=np.sum(transition_prob[t_idx,m_idx,b,:])
        m_idx+=1
    mice_mean_transition_prob[t_idx,:,:] = np.sum(transition_prob[t_idx,:,:,:],axis=0) / float(num_of_mice[t_idx])
    mice_mean_time_spent[t_idx,:] = np.sum(time_spent[t_idx,:,:], axis=0) / float(num_of_mice[t_idx])
    t_idx+=1

print('Done collecting data')
mice_mean_time_spent*=1000
transition_threshold = 1.0/(num_of_behaviors**2) # randomly selecting behavior and randomly selecting to move to the nexet behavior
aboveChance_alpha=0.8
belowChance_alpha=0.12
for t_idx in range(num_of_subset_trials):
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
    plt.title(trials_subset[t_idx]+'\n$\pi$[b$_t$,b$_{t+1}$]')
    nx.draw(G, pos=nx.circular_layout(G),edge_color=edge_colors, node_color=colors,width=list(weights),node_size=mice_mean_time_spent[t_idx],arrowstyle='->'
            ,arrowsize=15 , connectionstyle='arc3, rad = 0.1',with_labels=False)
    plt.savefig(output_folder+'states_actions_'+trials_subset[t_idx]+'.png',dpi=300)
    plt.close()

saline_transiton_prob = np.nanmean(mice_mean_transition_prob[:3,:,:],axis=0)
saline_time_spent = np.nanmean(mice_mean_time_spent[:3, :], axis=0)
plt.figure()
edges = []
edge_colors = []
edge_width = []
for b_cur in range(num_of_behaviors):
    for b_next in range(num_of_behaviors):
        edges.append((b_cur, b_next, saline_transiton_prob[b_cur, b_next] * 5))
        if saline_transiton_prob[b_cur, b_next] >= transition_threshold:
            edge_colors.append([0.294, 0, 0.51, aboveChance_alpha])
        else:
            edge_colors.append([0.294, 0, 0.51, belowChance_alpha])
G = nx.DiGraph()
G.add_weighted_edges_from(edges, color='red')
weights = nx.get_edge_attributes(G, 'weight').values()
plt.title('Mean Saline days\n$\pi$[b$_t$,b$_{t+1}$]')
nx.draw(G, pos=nx.circular_layout(G), edge_color=edge_colors, node_color=colors, width=list(weights), node_size=saline_time_spent,
        arrowstyle='->'
        , arrowsize=15, connectionstyle='arc3, rad = 0.1', with_labels=False)
plt.savefig(output_folder + 'states_actions_mean_saline_days.png', dpi=300)
plt.close()


diff_transition_prob = mice_mean_transition_prob[3:,:,:]-saline_transiton_prob
for t_idx in np.arange(num_of_subset_trials-3):
    plt.figure()
    edges = []
    edge_colors = []
    edge_width = []
    for b_cur in range(num_of_behaviors):
        for b_next in range(num_of_behaviors):
            edges.append((b_cur, b_next, diff_transition_prob[t_idx, b_cur, b_next] * 20))
            if diff_transition_prob[t_idx, b_cur, b_next] >= transition_threshold :
                color = [0.29, 0, 0.75,aboveChance_alpha]
            elif diff_transition_prob[t_idx, b_cur, b_next] <= -transition_threshold:
                color = [1, 0,0,aboveChance_alpha]
            elif diff_transition_prob[t_idx, b_cur, b_next] <= transition_threshold and diff_transition_prob[t_idx, b_cur, b_next] >0:
                color = [0.29, 0, 0.75,belowChance_alpha]
            else:
                color =[1, 0,0 ,belowChance_alpha]
            edge_colors.append(color)
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges, color='red')
    weights = nx.get_edge_attributes(G, 'weight').values()
    plt.title(trials_subset[t_idx+3]+'\n$\Delta$ $\pi$[b$_t$,b$_{t+1}$]' )
    nx.draw(G, pos=nx.circular_layout(G), edge_color=edge_colors, node_color=colors, width=list(weights), node_size=mice_mean_time_spent[t_idx+3],
            arrowstyle='->', arrowsize=15, connectionstyle='arc3, rad = 0.1', with_labels=False)
    plt.savefig(output_folder + 'states_actions_diff_' + trials_subset[t_idx+3] + '.png', dpi=300)
    plt.close()

#%% CDFs pf bout length
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = root_folder+'/Figures/Stereotypies_development/Bouts/'
colors_prime = ['gray','gray','gray','blue','turquoise','green','orange','red']
alpha_prime = [.3,.5,.7,.5,.5,.5,.5,.5]
longest_bout = 400
x_axis = np.arange(longest_bout)
num_of_mice = 0
for t in trials:
    num_of_mice = np.max([num_of_mice,len(TM[t].keys())])
print(num_of_mice)
cdf = np.empty((num_of_behaviors,num_of_trials,num_of_mice,longest_bout))
cdf[:,:,:,:]=np.nan

for b in range(num_of_behaviors):
    plt.figure()
    tr_idx = 0
    for tr in trials:
        num_of_bouts = []
        bout_length = []
        m_idx=0
        for m in TM[tr]:
            predictions = TM[tr][m]['merged']
            bouts_data = np.array(hf.segment_bouts(predictions,b,8)['length'])
            if bouts_data.size>0:
                for bl in x_axis:
                    cdf[b,tr_idx,m_idx,bl] = np.count_nonzero(bouts_data<=bl)/bouts_data.size
            m_idx+=1
        mean = np.nanmean(cdf[b,tr_idx,:,:],axis=0)
        stderr = np.sqrt(np.nanvar(cdf[b,tr_idx,:,:],axis=0)/m_idx)
        plt.plot(x_axis,mean,color = colors_prime[tr_idx],alpha=.7,label = tr)
        plt.fill_between(x_axis,y1=mean-stderr,y2=mean+stderr, color=colors_prime[tr_idx], alpha=.3)
        tr_idx+=1
    plt.xlabel('Bout length (# of frames)')
    plt.ylabel('${P}(|bout| \leq {x}$)')
    plt.title('CDF of bout length for behavior: '+behaviors[b])
    plt.legend()
    sns.despine()
    plt.savefig(output_folder+'bout_length_cdf'+behaviors[b]+'.pdf',dpi=300)
    plt.close()
#%%Change in velocity for floor licking, across the days of stereotypies development
output_folder = root_folder+'/Figures/Stereotypies_development/Movement_parameters/'
x_axis = np.arange(0,0.01,1e-5)*FPS
vel = {}
behaviors_of_interset = [ FL ]
for b in behaviors_of_interset:
    vel[b] = {}
    for tr in trials:
        vel[b][tr] = []
for tr in trials:
    for m in TM[tr]:
        predictions = TM[tr][m]['merged']
        session_vel = TM[tr][m]['topcam']['velocity']*FPS
        session_vel = hf.remove_IQR_outliers(session_vel)
        for b in behaviors_of_interset:
            vel[b][tr].extend(session_vel[predictions==b])
print('Finshed collecting data')
colors_prime = ['gray','gray','gray','blue','turquoise','green','orange','red']
alpha_prime = [.3,.5,.7,.5,.5,.5,.5,.5]
for b in behaviors_of_interset:
    print('Plotting stat for ',behaviors[b])
    fig , ax = plt.subplots(ncols = 2,figsize=(14,6))
    plt.sca(ax[0])
    tr_idx = 0
    for tr in trials:
        print('Hist')
        sns.histplot(vel[b][tr],label = tr,alpha=alpha_prime[tr_idx],stat = 'probability',color = colors_prime[tr_idx],bins=100)
        tr_idx += 1
    plt.xlabel('vel (m/s)')
    plt.title('Historgam')
    plt.legend()
    plt.sca(ax[1])
    tr_idx = 0
    for tr in trials:
        print('CDF')
        if len(vel[b][tr])>0:
            b_tr_cdf = []
            for x in x_axis:
                b_tr_cdf.append(np.count_nonzero(vel[b][tr] <= x) / len(vel[b][tr]))
            plt.plot(x_axis,b_tr_cdf,c = colors_prime[tr_idx],alpha = alpha_prime[tr_idx],label = tr)
        tr_idx+=1
    plt.xlabel('vel (m/s)')
    plt.title('CDF')
    plt.legend()
    plt.suptitle('Velocity stat across the days of the development of stereotypies\n Behavior : '+behaviors[b])
    plt.savefig(output_folder+'/vel_stat_'+behaviors[b]+'.png',dpi=300)
    plt.close()


