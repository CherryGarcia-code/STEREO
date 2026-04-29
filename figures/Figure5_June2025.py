import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import copy
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
from numpy.linalg import norm
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
FPS=15
# ***CHANGE THE COHORT HERE***
cohort = 'a2a_hm3dq'
MT = pickle.load(ifile)[cohort]
ifile.close()
CNO2cocaineGap = {'drd1_hm4di':{'c512m3':30,'c512m4':30,'c512m7':30,'c526m2':32,'c526m3':35,'c528m5':31,'c528m10':38,'c548m1':31},
                  'drd1_hm3dq':{'c514Bm2':35,'c514Bm8':35,'c514m1':35,'c514m3':38,'c514m5':32},
                  'controls':{'c548m8':33,'c548m10':32,'c548m11':32,'cA242m4':30,'cA242m9':30},
                  'a2a_hm4di':{'cA154m4':35,'cA154m6':36,'cA156m1':35,'cA156m6':35,'cA156m7':35,'cA156m8':34},
                  'a2a_hm3dq':{'cA156m2':33,'cA156m5':34,'cA158m2':30,'cA158m3':30,'cA158m4':30,'cA184m4':33,'cA184m7':33,'cA242m5':30,'cA242m6':30,'cA242m8':30}}
possible_trials= {'cocaineOnly':['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5','cocaine6','cocaine7','cocaine8','cocaine9','cocaine10'],
                                'cocaineCNO':['cocaine6afterCNO','cocaine7afterCNO', 'cocaine8afterCNO', 'cocaine9afterCNO']}
gcamp_mice = ['c528m5', 'c528m10', 'c548m1', 'c514Bm2', 'c514Bm8', 'cA184m4', 'cA184m7', 'cA242m5', 'cA242m6','cA242m8']
subOptimal_infection = ['c512m4']
lut = np.array([4,5,3,2,6,1,8,7,0])
sandwich_days = {}
for m in MT.keys():
    if m in gcamp_mice or m in subOptimal_infection:continue
    trials = list(MT[m].keys())
    for t_idx in range(1, len(trials) - 1):
        if trials[t_idx] in possible_trials['cocaineCNO'] and trials[t_idx + 1] in possible_trials['cocaineOnly'] and \
                (trials[t_idx - 1] in possible_trials['cocaineOnly'] or ('CNOonly' in trials[t_idx - 1] and trials[t_idx - 2] in possible_trials['cocaineOnly'])):
            sandwich_days[m]={}
            sandwich_days[m]['CNO'] = {'behavior':np.copy(MT[m][trials[t_idx]]['merged']['predictions']['smartMerge']),
                                       'velocity':np.copy(MT[m][trials[t_idx]]['topcam']['velocity'])*FPS*10}
            sandwich_days[m]['post'] = {'behavior':np.copy(MT[m][trials[t_idx+1]]['merged']['predictions']['smartMerge']),
                                        'velocity':np.copy(MT[m][trials[t_idx+1]]['topcam']['velocity'])*FPS*10}

            if trials[t_idx - 1] in possible_trials['cocaineOnly']:
                print(m, trials[t_idx-1],trials[t_idx],trials[t_idx+1])
                pre_tr = trials[t_idx-1]
                sandwich_days[m]['pre'] = {
                    'behavior': np.copy(MT[m][trials[t_idx - 1]]['merged']['predictions']['smartMerge']),
                    'velocity': np.copy(MT[m][trials[t_idx - 1]]['topcam']['velocity'])*FPS*10}
            else:
                print(m, trials[t_idx - 2], trials[t_idx], trials[t_idx + 1])
                pre_tr = trials[t_idx - 2]
                sandwich_days[m]['pre'] = {
                    'behavior': np.copy(MT[m][trials[t_idx - 2]]['merged']['predictions']['smartMerge']),
                    'velocity': np.copy(MT[m][trials[t_idx - 2]]['topcam']['velocity'])*FPS*10}

            for tr in ['cocaine1', 'cocaine2', 'cocaine3', 'cocaine4', 'cocaine5']:
                if tr == pre_tr:
                    nan_arr = np.empty(sandwich_days[m]['pre']['behavior'].shape)
                    nan_arr[:]=np.nan
                    sandwich_days[m][tr] = {'behavior': nan_arr,'velocity': None}
                else:
                    sandwich_days[m][tr] = {'behavior': np.copy(MT[m][tr]['merged']['predictions']['smartMerge']),
                        'velocity': np.copy(MT[m][tr]['topcam']['velocity']) * FPS * 10}
            break

del MT
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
FPS=15
sample_rate=15
second = 15
minute = 60*second
ISI = 1 * minute + 20*second
CNO_MAX_TIME = 50
PRE = 0
DURING = 1
POST = 2
num_of_days = 3
num_of_behaviors = len(behaviors)
RNN_offset = 7
locomotive = 0
stationary = 1
cutoff = 15*minute
num_of_mice = len(list(sandwich_days.keys()))
PATHO_LICKING = 0
NATURAL_LICKING = 1
NO_LICKING = 2
grouping_lut =  np.array([NO_LICKING,NO_LICKING,PATHO_LICKING,PATHO_LICKING,NATURAL_LICKING,NATURAL_LICKING,NO_LICKING,NO_LICKING,NO_LICKING,NO_LICKING])
num_of_grouped_behaviors=3
UNDEFINED=1
FLOOR_LICKING = 2
WALL_LICKING = 3
LOCOMOTION = 7
OTHER = np.array([0,4,5,6,8])
days = ['Pre-CNO','During-CNO','Post-CNO','Pre-CNO']
mm=1/25.4
#%% Fig 5E,K
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Ethograms/'
for m in sandwich_days:
    cutoff = (CNO_MAX_TIME - CNO2cocaineGap[cohort][m]) * minute
    fig , ax = plt.subplots(nrows=3,sharex='all',figsize=(130*mm,40*mm))
    pre = np.copy(sandwich_days[m]['pre']['behavior'])[:cutoff]
    CNO = np.copy(sandwich_days[m]['CNO']['behavior'])[:cutoff]
    post = np.copy(sandwich_days[m]['post']['behavior'])[:cutoff]
    pre = np.reshape(pre, (1, pre.shape[0]))
    CNO = np.reshape(CNO, (1, CNO.shape[0]))
    post = np.reshape(post, (1, post.shape[0]))
    ax[0] = sns.heatmap(pre, yticklabels=[''], cmap=colors,cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[0])
    ax[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[0].tick_params(axis='y', which='both', left=False)
    ax[1] = sns.heatmap(CNO, yticklabels=[''], cmap=colors,cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[1])
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[1].tick_params(axis='y', which='both', left=False)
    ax[2] = sns.heatmap(post, yticklabels=[''], cmap=colors,cbar = False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[2])
    # ax[2].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax[2].tick_params(axis='y', which='both', left=False)
    plt.sca(ax[2])
    plt.xticks([0, 2700], [],rotation=0)
    plt.xlabel('Time from cocaine (m)',fontsize=10)
    plt.gca().tick_params(axis='y', which='both', left=False)
    fig.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0.1)
    plt.savefig(output_folder+m+'_woVel.png',dpi=300)
    plt.savefig(output_folder + m +'_woVel.pdf', dpi=300)
    plt.close()
#%% Fig 5F,L
output_folder = root_folder+'/Figures/DREADDS/'+cohort+'/Summary'
fraction_of_frames = np.zeros((num_of_mice , num_of_behaviors , num_of_days))
clustered_fraction_of_frames = np.zeros((num_of_mice, 3 , num_of_days))
UNDEFINED = 1
m_idx = 0
for m in sandwich_days:

    cutoff = (CNO_MAX_TIME - CNO2cocaineGap[cohort][m]) * minute
    pre = np.copy(sandwich_days[m]['pre']['behavior'])[:cutoff]
    CNO = np.copy(sandwich_days[m]['CNO']['behavior'])[:cutoff]
    post = np.copy(sandwich_days[m]['post']['behavior'])[:cutoff]
    pre = pre[pre!=UNDEFINED] # excluding BTC frames from analysis
    CNO = CNO[CNO != UNDEFINED] # excluding BTC frames from analysis
    post = post[post != UNDEFINED] # excluding BTC frames from analysis
    for b in range(num_of_behaviors):
        try:
            fraction_of_frames[m_idx , b , PRE] = np.count_nonzero(pre==b)/np.count_nonzero(pre==b)
            fraction_of_frames[m_idx, b, DURING] = np.count_nonzero(CNO == b) / np.count_nonzero(pre==b)
            fraction_of_frames[m_idx, b, POST] = np.count_nonzero(post == b) / np.count_nonzero(pre==b)
        except ZeroDivisionError:
            continue
    clustered_pre = grouping_lut[pre]
    clustered_CNO = grouping_lut[CNO]
    clustered_post = grouping_lut[post]
    for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
        clustered_fraction_of_frames[m_idx,category,PRE] = np.count_nonzero(clustered_pre==category)/ pre.size
        clustered_fraction_of_frames[m_idx, category, DURING] = np.count_nonzero(clustered_CNO == category) /  CNO.size
        clustered_fraction_of_frames[m_idx, category, POST] = np.count_nonzero(clustered_post == category) /  post.size
    m_idx+=1
for b in range(num_of_behaviors):
    plt.figure(frameon=False,figsize=(40*mm,40*mm))
    mean = np.mean(fraction_of_frames[:,b,:],axis=0)
    stderr = np.sqrt(np.var(fraction_of_frames[:, b, :], axis=0)/num_of_mice)
    plt.plot(fraction_of_frames[:,b,:].T, alpha=0.3, color='gray', marker='o',lw=.5,markersize=1)
    plt.plot(mean.T, color=colors[b],lw=.7)
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=colors[b],capsize=2,capthick=.5,lw=.7)
    plt.xticks(np.arange(3),['Pre-CNO','During-CNO','Post-CNO'])
    plt.ylabel('Fraction of frames')

    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.title('Change in behavior frequency around CNO day (first '+str(int(cutoff//minute))+'minutes)\n Behaviors: '+behaviors[b])
    plt.savefig(output_folder + '/V_graph_'+behaviors[b]+ '.png', dpi=300)
    plt.close()

category2name = ['Surface-Licking','Self-Directed-Licking ','No-licking']
category2colors = ["#d73027", "#c4a7e7", "#1f4e79"]
for category in [PATHO_LICKING,NATURAL_LICKING,NO_LICKING]:
    plt.figure(frameon=False,figsize=(40*mm,40*mm))
    mean = np.mean(clustered_fraction_of_frames[:,category,:],axis=0)
    stderr = np.sqrt(np.var(clustered_fraction_of_frames[:,category,:], axis=0)/num_of_mice)
    plt.plot(clustered_fraction_of_frames[:,category,:].T, alpha=0.3, color='gray', marker='o',lw=1,markersize=2)
    plt.plot(mean, color=category2colors[category],lw=1)
    plt.errorbar(np.arange(3),y=mean.T,yerr=stderr.T,color=category2colors[category],capsize=2,capthick=1,lw=.7)
    plt.xticks(np.arange(3),['-','+','-'],fontsize=8)
    plt.xlabel('CNO',fontsize=10)
    plt.ylabel('% Time spent',fontsize=10)
    plt.yticks([0.2,0.6,1],[20,60,100],fontsize=8)
    plt.ylim([0.2,1])
    axes = plt.gca()
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0.1)
    plt.savefig(output_folder + '/V_graph_'+category2name[category].replace('-','_')+'.png', dpi=300)
    plt.savefig(output_folder + '/V_graph_' + category2name[category].replace('-', '_') + '.pdf', dpi=300)
    plt.close()
print('Floor licking')
print('PRE VS CNO : ', stats.ttest_rel(fraction_of_frames[:,FLOOR_LICKING,PRE],fraction_of_frames[:,FLOOR_LICKING,DURING],alternative='greater'))
print('CNO VS POST : ', stats.ttest_rel(fraction_of_frames[:,FLOOR_LICKING,DURING],fraction_of_frames[:,FLOOR_LICKING,POST],alternative='less'))
print('PRE VS POST : ' , stats.ttest_rel(fraction_of_frames[:,FLOOR_LICKING,PRE],fraction_of_frames[:,FLOOR_LICKING,POST]))
print('Wall licking')
print('PRE VS CNO : ', stats.ttest_rel(fraction_of_frames[:,WALL_LICKING,PRE],fraction_of_frames[:,WALL_LICKING,DURING],alternative='less'))
print('CNO VS POST : ', stats.ttest_rel(fraction_of_frames[:,WALL_LICKING,DURING],fraction_of_frames[:,WALL_LICKING,POST],alternative='greater'))
print('PRE VS POST : ' , stats.ttest_rel(fraction_of_frames[:,WALL_LICKING,PRE],fraction_of_frames[:,WALL_LICKING,POST]))
if cohort in ['a2a_hm4di','drd1_hm3dq']:
    direction = ['less', 'greater']
else:
    direction = ['greater','less']
print(cohort , ' ; Surface licking')
print('2-tailed')
print('2-Sided : PRE VS CNO : ', stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,PRE],clustered_fraction_of_frames[:,PATHO_LICKING,DURING]))
print('2-Sided : CNO VS POST : ', stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,DURING],clustered_fraction_of_frames[:,PATHO_LICKING,POST]))
print('2-Sided : PRE VS POST : ' , stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,PRE],clustered_fraction_of_frames[:,PATHO_LICKING,POST]))
print('1-tailed')
print('1-Sided : PRE VS CNO : ', stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,PRE],clustered_fraction_of_frames[:,PATHO_LICKING,DURING],alternative=direction[0]))
print('1-Sided : CNO VS POST : ', stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,DURING],clustered_fraction_of_frames[:,PATHO_LICKING,POST],alternative=direction[1]))
print('1-Sided : PRE VS POST : ' , stats.ttest_rel(clustered_fraction_of_frames[:,PATHO_LICKING,PRE],clustered_fraction_of_frames[:,PATHO_LICKING,POST],alternative='less'))
#%% Fig 5G,H
cutoff=15*minute
mpl.rcParams["mathtext.fontset"] = 'cm'
output_folder = 'output/DREADDS/'+cohort+'/Bouts/'
labels_dict = {PRE:'Pre CNO',DURING:'CNO',POST:'Post CNO'}
longest_bout = 150
x_axis = np.arange(0,longest_bout,1)
cdf = np.empty((num_of_behaviors,3,num_of_mice,x_axis.size))
cdf[:,:,:,:]=np.nan
for b in range(num_of_behaviors):
    plt.figure(frameon=False,figsize=(2,1.9))
    m_idx=0
    for m in sandwich_days:
        preCNO = np.copy(sandwich_days[m]['pre']['behavior'][:cutoff])
        CNO = np.copy(sandwich_days[m]['CNO']['behavior'][:cutoff])
        postCNO = np.copy(sandwich_days[m]['post']['behavior'][:cutoff])
        tr_idx=0
        for predictions in [preCNO,CNO,postCNO]:
            bouts_data = np.array(hf.segment_bouts(predictions,b,8)['length'])
            if bouts_data.size>0:
                bl_idx=0
                for bl in x_axis:
                    cdf[b,tr_idx,m_idx,bl_idx] = np.count_nonzero(bouts_data<=bl)/bouts_data.size
                    bl_idx+=1
            tr_idx+=1
        m_idx+=1

    colors_prime = ['gray', colors[b], 'gray']
    alpha_prime = [.3,1,.7]
    for tr_idx in range(3):
        mean = np.nanmean(cdf[b,tr_idx,:,:],axis=0)
        stderr = np.sqrt(np.nanvar(cdf[b,tr_idx,:,:],axis=0)/m_idx)
        plt.plot(x_axis,mean,color = colors_prime[tr_idx],alpha = alpha_prime[tr_idx],label = labels_dict[tr_idx],lw=.7)
        plt.fill_between(x_axis,y1=mean-stderr,y2=mean+stderr, color=colors_prime[tr_idx], alpha=.3)
    plt.xticks(np.arange(0,151,15),np.arange(11),fontsize=8)
    plt.xlim([0,150])
    plt.ylim([0, 1])
    plt.xlabel('Bout length(s)',fontsize=10)
    plt.ylabel('CDF',fontsize=10)
    plt.yticks([0,.5,1],[0,.5,1])
    plt.title('Bout duration',fontsize=12)
    plt.legend(frameon=False,fontsize=8)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=0.1)
    plt.savefig(output_folder+'bout_length_cdf'+behaviors[b]+'.png',dpi=300)
    plt.savefig(output_folder + 'bout_length_cdf' + behaviors[b] + '.pdf', dpi=300)
    plt.close()

#%% Fig 5I,J
FL = 2
LOC = 7
behaviors_of_interest = [FL , LOC]
vel = {'pre':{FL:[] , LOC:[]} , 'CNO':{FL:[] , LOC:[]},'post':{FL:[] , LOC:[]}}
for b in behaviors_of_interest:
    for m in sandwich_days:
        preCNO = np.copy(sandwich_days[m]['pre']['behavior'][:cutoff])
        pre_vel = np.copy(sandwich_days[m]['pre']['velocity'][:cutoff])
        CNO = np.copy(sandwich_days[m]['CNO']['behavior'][:cutoff])
        CNO_vel = np.copy(sandwich_days[m]['CNO']['velocity'][:cutoff])
        postCNO = np.copy(sandwich_days[m]['post']['behavior'][:cutoff])
        post_vel = np.copy(sandwich_days[m]['post']['velocity'][:cutoff])
        vel_b = pre_vel[preCNO==b]
        vel_b = vel_b[hf.remove_IQR_outliers(vel_b)]
        vel['pre'][b].extend(vel_b)
        vel_b = CNO_vel[CNO == b]
        vel_b = vel_b[hf.remove_IQR_outliers(vel_b)]
        vel['CNO'][b].extend(vel_b)
        vel_b = post_vel[postCNO == b]
        vel_b = vel_b[hf.remove_IQR_outliers(vel_b)]
        vel['post'][b].extend(vel_b)

    vel['pre'][b] = np.array(vel['pre'][b])[~np.isnan(vel['pre'][b])]
    vel['CNO'][b] = np.array(vel['CNO'][b])[~np.isnan(vel['CNO'][b])]
    vel['post'][b] = np.array(vel['post'][b])[~np.isnan(vel['post'][b])]
axis_idx=0
for b in behaviors_of_interest:
    plt.figure(figsize=(2,2),frameon=False)
    sns.histplot(vel['pre'][b],color = 'gray',alpha = .3,stat = 'probability',label = 'Pre CNO',bins=100)
    sns.histplot(vel['CNO'][b], color=colors[b], stat='probability', label='During CNO', bins=100)
    sns.histplot(vel['post'][b], color='gray', alpha=.7, stat='probability', label='Post CNO', bins=100)
    plt.legend()
    plt.xlabel('Velocity(cm/s)',fontsize=10)
    plt.ylabel('Probability',fontsize=10)
    plt.legend(frameon=False,fontsize=8,loc='upper right')
    plt.title('Velocity',fontsize=12)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.yticks([0,0.05,0.1],[0,0.05,0.1],fontsize=8)
    plt.xticks([0, 0.5, 1], [0, .5, 1], fontsize=8)
    plt.xlim(0,1)
    plt.tight_layout(rect=[0.01, 0.01, .99, 0.99], pad=.1)
    plt.savefig(output_folder+'/vel_stat_'+behaviors[b]+'.png',dpi=300)
    plt.savefig(output_folder + '/vel_stat_' + behaviors[b] + '.pdf', dpi=300)
    plt.close()
    axis_idx+=1
