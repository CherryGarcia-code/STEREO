#%% Imports and constants
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as hf
import bz2
import pickle
import copy
import sys
import os
import stats_helper_file as shf
mm=1/25.4
FPS=15
sample_rate=15
second = 15
minute = 60*second
#%% Loading saline data
sys.modules[__name__].__dict__.clear()
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
trials  = ['saline1','saline2','saline3','splashTest']
cohorts = ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq','a2a_opto']
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
num_of_trials = len(trials)
num_of_behaviors = len(behaviors)
MT = {}
TM = {}
for c in cohorts:
    for m in CMT[c]:
        MT[m] = copy.deepcopy(CMT[c][m])
for cohort in cohorts:
    for mouse in CMT[cohort].keys():
        for t in CMT[cohort][mouse].keys():
            if t not in TM.keys():
                TM[t] = {}
            TM[t][mouse] = copy.deepcopy(CMT[cohort][mouse][t])
del CMT
#%% Fig 1D
mouse = 'c514m5'
trial = 'saline1'
data = MT[mouse][trial]
predictions = data['merged']['predictions']['smartMerge']
predictions = np.reshape(predictions,(1,predictions.size))
locomotion = data['topcam']['velocity']*FPS*10
SSD = data['SSD']
SSD = np.max([SSD['cam1'],SSD['cam1']],axis=0)
SSD = hf.smoothing(SSD,5)*1000
fig, ax, = plt.subplots(3, sharex=True, figsize=(4.5,3.5),frameon=False)
ax[0] = sns.heatmap(predictions, yticklabels=[''],cbar=False, cmap=colors, vmin=0, vmax=8, ax=ax[0])
ax[0].set_ylabel('Predictions', va='center',fontsize = 10)
ax[1].plot(locomotion,lw=.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.sca(ax[1])
plt.ylabel('Velocity',fontsize=10)
plt.yticks([0,1],['0','1'],fontsize=8)
ax[2].plot(SSD, alpha=.5,lw=.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.sca(ax[2])
plt.ylabel('Put. vigor',fontsize=10)
plt.yticks([0,1],['0','1'],fontsize=8)
plt.xticks([0,4500], [0,5], rotation=0, fontsize=8)
plt.savefig('output/Fig1_salineAnalysis/rep_ethogram.pdf', bbox_inches='tight',dpi=300)
#%% Fig 1E
snippets_folder = 'C:/Users/owner/Desktop/Itay/PhD/Confrences/ISFN 24/Code/Data/STEREO/files/'
output_folder = 'output/STEREO/'
files_list = sorted(os.listdir(snippets_folder))
print(files_list)
num_of_snippets=14*3
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other']
snippets_lut = np.array([3,4,2,1,5,0,6])
colors_subset = ["#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc"]
behaviors_remapped = ['Undefined','Floor licking','Wall licking','Grooming','Body licking','Rearing','Loco/Stat']
for i in range(0,num_of_snippets,2):
    print(snippets_folder+files_list[i],snippets_folder+files_list[i+1],snippets_folder+files_list[i+2])
    H1 = np.load(snippets_folder+files_list[i]).astype(np.int16)
    print(files_list[i],files_list[i+1])
    STEREO = np.load(snippets_folder+files_list[i+1]).astype(np.int16)
    H1 = snippets_lut[H1]
    STEREO = snippets_lut[STEREO]
    H1 = np.reshape(H1,(1,H1.size))[:,7:STEREO.size+7]
    STEREO = np.reshape(STEREO,(1,STEREO.size))
    print([H1.size,STEREO.size])
    ethogram = np.vstack([STEREO,H1])
    plt.figure(figsize=(54*mm,23*mm),frameon=False)
    ax = sns.heatmap(ethogram, cbar=False, cmap=colors_subset,
                     cbar_kws={'ticks': np.arange(len(behaviors)),
                               'boundaries': np.arange(-1, len(behaviors)) + 0.5,
                               'drawedges': True}, vmin=0, vmax=6)
    ax.spines[:].set_visible(False)
    ax.set_yticks([1], minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)  # change the appearance of your padding here
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.yticks([0.5,1.5],['',''],fontsize=10,rotation=0)
    plt.xticks([900],'')
    plt.tight_layout(rect=[0.01, 0.05, 0.95, 0.95], pad=.1)

    plt.savefig(output_folder + 'comparative_ethograms_' + files_list[i][:-11] + '.png', bbox_inches='tight',
                dpi=300)
    plt.savefig(output_folder + 'comparative_ethograms_' + files_list[i][:-11] + '.pdf', bbox_inches='tight',
                dpi=300)
    plt.close()
#%% Fig 1F
mat = np.array([[80,1,0,0,0,5,14],[3,83,0,0,0,12,2],[0,0,85,0,1,11,3],[0,0,2,81,0,10,6],[0,1,0,0,84,11,4],[8,1,1,0,1,81,7],[1,0,0,0,0,3,95]])
mat = mat[::-1,::-1]
plt.figure(figsize=(47*mm,47*mm),frameon=False)
ax = sns.heatmap(mat,annot=True,cmap='Purples',cbar = False,annot_kws={"size":8},square=True)
plt.ylabel('Observer 1',fontsize=10)
plt.xlabel('Observer 2',fontsize=10)
plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
plt.tick_params(axis='y',which='both',right=False,left=False,labelleft=False)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.tight_layout(rect=[0.01, 0.01, .99, .99],pad=0)
plt.savefig('output/Fig1_salineAnalysis/confusionMatrix_H1XH2.pdf',dpi=300,bbox_inches='tight')
plt.close()
#%% Fig 1G
mat = np.array([[63,8,0,0,0,21,9],[1,92,0,0,1,3,3],[0,1,88,1,0,8,2],[0,1,2,77,0,5,15],[0,12,0,0,72,9,7],[4,2,2,1,2,83,6],[3,3,0,0,0,6,87]])
mat = mat[::-1,::-1]
plt.figure(figsize=(47.5*mm,47.5*mm),frameon=False)
ax = sns.heatmap(mat,annot=True,cmap='Purples',cbar = False,annot_kws={"size":8},square=True)
plt.ylabel('Observer 1',fontsize=10)
plt.xlabel('STEREO',fontsize=10)
plt.tick_params(axis='x',which='both',bottom=False,labelbottom=False)
plt.tick_params(axis='y',which='both',right=False,left=False,labelleft=False)
plt.gca().spines['right'].set_visible(True)
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)
plt.tight_layout(rect=[0.01, 0.01, .99, .99],pad=0)
plt.savefig('output/Fig1_salineAnalysis/confusionMatrix_H1XSTEREO.pdf',dpi=300,bbox_inches='tight')
plt.close()
#%% Fig 1H
init_flag = True
for t in trials[:-1]:
    for m in TM[t]:
        tm_dist = np.zeros(num_of_behaviors)
        for b in range(num_of_behaviors):
            predictions = TM[t][m]['merged']
            tm_dist[b] = np.count_nonzero(predictions==b)/predictions.shape[0]
        if init_flag:
            dist = np.copy(tm_dist)
            init_flag = False
        else:
            dist = np.vstack([dist,tm_dist])

mean = np.mean(dist,axis=0)
stderr = np.sqrt(np.var(dist,axis=0)/dist.shape[0])
plt.figure(figsize=(3,2),frameon= False)
plt.bar(np.arange(num_of_behaviors),mean,align='center',color = colors,width=.5,label=behaviors)
plt.legend(ncols=2,fontsize=8)
plt.ylabel('Frequency (%)',fontsize=10)
plt.xlabel('\nBehavior',fontsize= 10)
plt.tick_params(labelbottom = False)
plt.xticks(np.arange(num_of_behaviors))
plt.yticks([0.25,0.5,0.75],[25,50,75],fontsize=8)
plt.ylim(0,0.8)
plt.xlim(-.5,num_of_behaviors-.5)
for b in range(num_of_behaviors):
    plt.scatter(x=np.zeros(dist.shape[0])+b,y = dist[:,b],marker='o',edgecolors=colors[b],facecolors='white',alpha=.3)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.tight_layout(rect=[0.01, 0.05, 1, 0.95], pad=.1)
plt.savefig('output/Fig1_salineAnalysis/Distribution_wLegend_saline.pdf',dpi=300)
#%% Loading splash data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import helper_functions as hf
import bz2
import pickle
import stats_helper_file as shf
mm=1/25.4
FPS=15
sample_rate=15
second = 15
minute = 60*second
root_folder = '.'
folder = 'data/'
ifile = bz2.BZ2File(folder + 'CTM_Dec24.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT_Dec24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
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
del CTM
#%% Fig 1J
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
#%% Fig 1K
dist_grooming = []
dist_bl = []
GROOMING = 4
BODY_LICKING = 5
for m in splashTest_days:
    m_dist_grooming = []
    m_dist_bl = []
    bl1 = np.copy(splashTest_days[m]['pre1']['merged']['predictions']['smartMerge'])
    bl2 = np.copy(splashTest_days[m]['pre2']['merged']['predictions']['smartMerge'])
    bl3 = np.copy(splashTest_days[m]['pre3']['merged']['predictions']['smartMerge'])
    st = np.copy(splashTest_days[m]['splashTest']['merged']['predictions']['smartMerge'])
    for predictions in [bl1,bl2,bl3,st]:
        grooming_frac = np.count_nonzero(predictions==GROOMING)/predictions.size
        bl_frac = np.count_nonzero(predictions==BODY_LICKING)/predictions.size
        m_dist_grooming.append(grooming_frac)
        m_dist_bl.append(bl_frac)
    if len(m_dist_grooming)<4:
        continue
    else:
        dist_grooming.append(m_dist_grooming)
        dist_bl.append(m_dist_bl)


dist_grooming = np.array(dist_grooming)
dist_bl = np.array(dist_bl)
print(dist_grooming.shape)
mean_grooming = np.mean(dist_grooming,axis=0)
mean_bl = np.mean(dist_bl,axis=0)
x_axis_grooming = [0,4,8,12]
x_axis_bl = [1,5,9,13]
x_axis_selfLicking = [0,1,2,3]
plt.figure(figsize=(75*mm,50*mm),frameon= False)
# plt.bar(x_axis_grooming,mean_grooming,align='center',color = colors[GROOMING],width=.8,label='Grooming') # turn on for supplement figure
# plt.bar(x_axis_bl,mean_bl,align='center',color = colors[BODY_LICKING],width=.8,label='Body Licking') # turn on for supplement figure
plt.bar(x_axis_selfLicking,mean_grooming+mean_bl,align='center',edgecolor=colors[BODY_LICKING],facecolor = 'white',width=.4,label='Self-Directed Licking',hatch='//',)
t_idx=0
for x in x_axis_selfLicking:
    # plt.scatter(np.zeros(dist_grooming.shape[0])+x,dist_grooming[:,t_idx],marker='o',edgecolors=colors[GROOMING],facecolors='white',alpha=.3)# turn on for supplement figure
    # plt.scatter(np.zeros(dist_grooming.shape[0]) + x+1, dist_bl[:,t_idx],marker='o',edgecolors=colors[BODY_LICKING],facecolors='white',alpha=.3)# turn on for supplement figure
    plt.scatter(np.zeros(dist_grooming.shape[0])+x, dist_bl[:, t_idx]+dist_grooming[:, t_idx], marker='o',edgecolors=colors[BODY_LICKING], facecolors='white', alpha=.3)
    t_idx+=1
# plt.legend(fontsize=10,frameon=False,loc='upper left')# turn on for supplement figure
plt.ylabel('Time Spent (%)',fontsize=10)
plt.yticks([0.25,0.5],[25,50],fontsize=8)
plt.xticks(x_axis_selfLicking,['1','2','3','Sucrose\nsplash'],fontsize=8)
plt.ylim(0,0.5)
plt.xlim(-0.25,3.5)
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.tight_layout(rect=[0.01, 0.01, 1, 0.99], pad=.1)
plt.savefig('output/SplashTest_behaviorOnly/Distribution_wLegend_slashTestCopmparedtoSaline.png',dpi=300)
plt.savefig('output/SplashTest_behaviorOnly/Distribution_wLegend_slashTestCopmparedtoSaline.pdf',dpi=300)

"""Statistics"""
dist_sl = np.array(dist_bl)+np.array(dist_grooming)
shf.rmANOVA(dist_sl)
days = ['baseline1','baseline2','baseline3','sucrose splash']
for i in range(len(days)):
    for j in range(i+1,len(days)):
        print(f'Paired t-test : {days[i]} VS {days[j]}')
        shf.paired_ttest(dist_sl[:,i],dist_sl[:,j])
#%% Fig 1L,M
mm=1/25.4
output_folder = root_folder+'/Figures/SplashTest_behaviorOnly/Bouts/'
longest_bout = 150
shortest_bout=15
x_axis = np.arange(longest_bout)
for b in range(num_of_behaviors):
    baseline = []
    splash = []
    for m in splashTest_days:
        pre1 = np.copy(splashTest_days[m]['pre1']['merged']['predictions']['smartMerge'])
        pre2 = np.copy(splashTest_days[m]['pre2']['merged']['predictions']['smartMerge'])
        pre3 = np.copy(splashTest_days[m]['pre3']['merged']['predictions']['smartMerge'])
        st = np.copy(splashTest_days[m]['splashTest']['merged']['predictions']['smartMerge'])
        durations = []
        for predictions in [pre1,pre2,pre3]:
            durations.extend(hf.segment_bouts(predictions,b,shortest_bout)['length'])
        baseline.append(durations)
        splash.append(hf.segment_bouts(st,b,shortest_bout)['length'])
    plt.figure(frameon=False, figsize=(40 * mm, 55 * mm))
    try:
        baseline_mean,baseline_err = hf.calc_cdf(baseline,x_axis)
        baseline_mean[:shortest_bout]=np.nan
        baseline_err[:shortest_bout] = np.nan
        plt.plot(baseline_mean, color=colors[b], ls='--', lw=.7, label='Baseline')
        plt.fill_between(x=x_axis, y1=baseline_mean - baseline_err, y2=baseline_mean + baseline_err, color=colors[b],
                         alpha=.3)
    except ZeroDivisionError:
        print('No bouts found for behavior '+behaviors[b]+' on baseline days')
    try:
        splash_mean, splash_err = hf.calc_cdf(splash,x_axis)
        splash_mean[:shortest_bout]=np.nan
        splash_err[:shortest_bout] = np.nan
        plt.plot(splash_mean, color=colors[b], lw=.7, label='Sucrose\nsplash')
        plt.fill_between(x=x_axis, y1=splash_mean - splash_err, y2=splash_mean + splash_err, color=colors[b], alpha=.3)
    except ZeroDivisionError:
        print('No bouts found for behavior '+behaviors[b]+' on splash day')
    plt.xlabel('Bout duration(s)',fontsize=10)
    plt.ylabel('CDF',fontsize=10)
    plt.yticks([0,.5,1],[0,.5,1],fontsize=8)
    plt.xticks(np.array([0,2,4,6,8,10])*FPS,[0,2,4,6,8,10],fontsize=8)
    plt.ylim([0,1])
    plt.legend(frameon=False,fontsize=8)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.tight_layout(rect=[0.01, 0.01, 1, 0.99], pad=.1)
    plt.savefig(output_folder+'bout_length_cdf'+behaviors[b]+'.png',dpi=300)
    plt.savefig(output_folder+'bout_length_cdf'+behaviors[b]+'.pdf',dpi=300)
    plt.close()
"""Statistics"""
BODY_LICKING=5
GROOMING=4
for b in [BODY_LICKING,GROOMING]:
    baseline_cdf = []
    splash_cdf = []
    for m in splashTest_days:
        pre1 = np.copy(splashTest_days[m]['pre1']['merged']['predictions']['smartMerge'])
        pre2 = np.copy(splashTest_days[m]['pre2']['merged']['predictions']['smartMerge'])
        pre3 = np.copy(splashTest_days[m]['pre3']['merged']['predictions']['smartMerge'])
        st = np.copy(splashTest_days[m]['splashTest']['merged']['predictions']['smartMerge'])
        baseline_durations = []
        for predictions in [pre1, pre2, pre3]:
            baseline_durations.extend(hf.segment_bouts(predictions, b, shortest_bout)['length'])
        splash_durations  = hf.segment_bouts(st, b, shortest_bout)['length']
        baseline_cdf.append(hf.dist_to_cdf(baseline_durations,x_axis))
        splash_cdf.append(hf.dist_to_cdf(splash_durations,x_axis))
    print('Behavior : '+behaviors[b])
    shf.paired_cdf_permutation_test(baseline_cdf,splash_cdf)
#%% Fig 1N,O
SSD = {}
saline_trial = 'salineOnly'
for b in range(num_of_behaviors):
    SSD[b] = []
for c in CMT:
    for m in CMT[c]:
        if saline_trial not in CMT[c][m] or 'splashTest' not in CMT[c][m] or 'saline1' not in CMT[c][m]:continue
        for b in range(num_of_behaviors):
            predictions = CMT[c][m][saline_trial]['merged']['predictions']['smartMerge']
            m_ssd = np.nanmax(np.vstack([ CMT[c][m][saline_trial]['SSD']['cam1'],  CMT[c][m][saline_trial]['SSD']['cam2']]), axis=0)*1000
            b_ssd_sal= m_ssd[predictions==b]
            predictions = CMT[c][m]['splashTest']['merged']['predictions']['smartMerge']
            m_ssd = np.nanmax(np.vstack([CMT[c][m]['splashTest']['SSD']['cam1'],  CMT[c][m]['splashTest']['SSD']['cam2']]), axis=0)*1000
            b_ssd_st = m_ssd[predictions == b]
            SSD[b].append([np.nanmean(b_ssd_sal),np.nanmean(b_ssd_st)])

for b in range(num_of_behaviors):
    SSD[b]=np.array(SSD[b])
    print(SSD[b].shape)

for b in [4,5,6,7,8]:
    plt.figure(figsize=(30*mm,55*mm),frameon= False)
    plt.plot(SSD[b].T,c='gray',alpha=.3,lw=.7)
    plt.plot([0,1],[np.mean(SSD[b][:, 0]),np.mean(SSD[b][:, 1])],c=colors[b],lw=1.5)
    # plt.hlines(y=np.mean(SSD[b][:, 0]), xmin=-.15, xmax=.15, colors=colors[b])
    plt.hlines(y = np.mean(SSD[b][:,0]),xmin=-.15, xmax=.15,colors=colors[b])
    plt.hlines(y=np.mean(SSD[b][:, 1]), xmin=.85, xmax=1.15, colors=colors[b])
    plt.yticks([2,3.5,5],[2,3.5,5],fontsize=8)
    plt.ylim([2,5])
    plt.xticks([0,1],['Baseline','Splash'],fontsize=8)
    plt.ylabel('Put. vigor(a.u.)', fontsize=10)
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.tight_layout(rect=[0.01, 0.01, 0.96, 0.99], pad=.1)
    plt.savefig('output/SplashTest_behaviorOnly/Movement_parameters/'+behaviors[b]+'.pdf',dpi=300)
    plt.savefig('output/SplashTest_behaviorOnly/Movement_parameters/' + behaviors[b] + '.png', dpi=300)
    plt.close()
"""statistics"""
LOCOMOTION = 7
print('Paired t-test - grooming')
shf.paired_ttest(SSD[GROOMING][:,0],SSD[GROOMING][:,1])
print('Paired t-test - locomotion')
shf.paired_ttest(SSD[LOCOMOTION][:,0],SSD[LOCOMOTION][:,1])