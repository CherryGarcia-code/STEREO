#%%
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
import networkx as nx
root_folder = 'May24'
folder = root_folder+'/Data/'
ifile = bz2.BZ2File(folder + 'CTM_Aug24.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT_Aug24.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Jump', 'Undefined', 'Floor licking', 'Wall licking', 'Grooming', 'Body licking', 'Rearing', 'Locomotion', 'Stationary']
colors = ["#696969", "#d3d3d3", "#d73027", "#e57373", "#c4a7e7", "#8e63b8", "#b3e5fc", "#3399cc", "#1f4e79"]
lut = np.array([4,5,3,2,6,1,8,7,0])
FPS=15
sample_rate=15
second = 15
minute = 60*second
mpl.rcParams['xtick.labelsize']=10
mpl.rcParams['ytick.labelsize']=10
mpl.rcParams['legend.fontsize']=8
VDB = 0
nVDB = 1
trials  = ['saline1','saline2','saline3','cocaine1','cocaine2','cocaine3','cocaine4','cocaine5']
cohorts = ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq','a2a_opto']
num_of_trials = len(trials)
num_of_behaviors = len(behaviors)
MT = {}
TM = {}
for c in cohorts:
    for m in CMT[c]:
        MT[m] = copy.deepcopy(CMT[c][m])
        for t in MT[m]:
            MT[m][t]['merged'] =  lut[MT[m][t]['merged']]
for cohort in cohorts:
    for mouse in CMT[cohort].keys():
        for t in CMT[cohort][mouse].keys():
            if t not in TM.keys():
                TM[t] = {}
            TM[t][mouse] = copy.deepcopy(CMT[cohort][mouse][t])
            TM[t][mouse]['merged'] = lut[TM[t][mouse]['merged']]
del CMT
del CTM
#%% Ethograms of all mice for saline1-cocaine5 days
mice_list = list(MT.keys())
for m in MT:
    for t in trials:
        if t not in MT[m]:
            mice_list.remove(m)
            break
num_of_mice = len(mice_list)

fig,ax = plt.subplots(nrows =num_of_trials ,ncols = num_of_mice,figsize=(24,6),frameon=False)
m_idx = 0
for m in mice_list:
    tr_idx=0
    for tr in trials:
        print(m,tr)
        predictions = MT[m][tr]['merged']
        predictions = np.reshape(predictions, (1, predictions.shape[0]))
        ax[tr_idx,m_idx] = sns.heatmap(predictions, yticklabels=[], cmap=colors,
                           cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[tr_idx,m_idx])
        ax[tr_idx, m_idx].spines['left'].set_visible(False)
        ax[tr_idx, m_idx].spines['bottom'].set_visible(False)
        ax[tr_idx, m_idx].spines['top'].set_visible(False)
        ax[tr_idx, m_idx].spines['right'].set_visible(False)
        plt.sca(ax[tr_idx,m_idx])
        plt.tick_params(left=False,  labelleft=False,labelbottom=False, bottom=False)
        if m_idx ==0:
            plt.ylabel(trials[tr_idx],fontsize=8)
        tr_idx+=1

    m_idx+=1

plt.subplots_adjust(left=0.01, right=.99, top=.92, bottom=0.05)
plt.savefig('May24/Figures/Stereotypies_development/Ethograms/allMice.pdf', bbox_inches='tight',dpi=300)
plt.savefig('May24/Figures/Stereotypies_development/Ethograms/allMice.png', bbox_inches='tight',dpi=300)
plt.close()

#%% All ethograms for a single mouse for saline1-cocaine5 days
for n in range(0, 36):
    mouse_to_plot = n
    mice_list = list(MT.keys())
    for m in MT:
        for t in trials:
            if t not in MT[m]:
                mice_list.remove(m)
                break
    num_of_mice = len(mice_list)
    mouse = mice_list[mouse_to_plot]
    fig,ax = plt.subplots(nrows =num_of_trials ,figsize=(24,6),frameon=False)
    tr_idx=0
    for tr in trials:
        print(tr)
        # if tr in MT[m]:
        predictions = MT[mouse][tr]['merged']
        predictions = np.reshape(predictions, (1, predictions.shape[0]))
        ax[tr_idx] = sns.heatmap(predictions, yticklabels=[], cmap=colors,
                        cbar=False, vmin=0, vmax=num_of_behaviors - 1, ax=ax[tr_idx])
        ax[tr_idx].spines['left'].set_visible(False)
        ax[tr_idx].spines['bottom'].set_visible(False)
        ax[tr_idx].spines['top'].set_visible(False)
        ax[tr_idx].spines['right'].set_visible(False)
        plt.sca(ax[tr_idx])
        plt.tick_params(left=False,  labelleft=False,labelbottom=False, bottom=False)
        plt.ylabel(trials[tr_idx],fontsize=8)
        tr_idx+=1

    plt.subplots_adjust(left=0.01, right=.99, top=.92, bottom=0.05)
    plt.savefig(f'May24/Figures/Stereotypies_development/Ethograms/singleMouse{mouse}.pdf', bbox_inches='tight',dpi=300)
    plt.savefig(f'May24/Figures/Stereotypies_development/Ethograms/singleMouse{mouse}.png', bbox_inches='tight',dpi=300)
    plt.close()
# %%
