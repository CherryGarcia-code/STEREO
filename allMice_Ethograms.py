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
model = 'model1'
folder = 'Data/'+model+'/'
ifile = bz2.BZ2File(folder + 'CTM.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()
print('***All dictionaries were loaded***')
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other', 'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#B2B2B2',  '#b4a7d6']
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
        MT[m] = CMT[c][m]
for cohort in cohorts:
    for mouse in CMT[cohort].keys():
        for t in CMT[cohort][mouse].keys():
            if t not in TM.keys():
                TM[t] = {}
            TM[t][mouse] = CMT[cohort][mouse][t]
del CMT
del CTM

num_of_mice = []
for t in trials:
    num_of_mice.append(len(TM[t].keys()))
max_num_of_mice = np.max(num_of_mice)
output_folder = 'Figures/'+model+'/Stereotypies_development/Ethograms/'
fig,ax = plt.subplots(nrows =num_of_trials ,ncols =max_num_of_mice,figsize=(24,6))
m_idx = 0
cbar_ax = fig.add_axes([.3, .92, .4, .02])
for m in MT:
    tr_idx=0
    for tr in trials:
        print(m,tr)
        if tr in MT[m]:
            predictions = MT[m][tr]['merged']
            predictions = np.reshape(predictions, (1, predictions.shape[0]))
            ax[tr_idx,m_idx] = sns.heatmap(predictions, yticklabels=[], cmap=colors,
                                cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5, 'drawedges': True,"orientation": "horizontal" },
                                cbar_ax=cbar_ax, vmin=0, vmax=num_of_behaviors - 1, ax=ax[tr_idx,m_idx])
            # plt.xticks(np.arange(predictions.shape[1]), "")
            ax[tr_idx, m_idx].spines['left'].set_visible(False)
            ax[tr_idx, m_idx].spines['bottom'].set_visible(False)
            ax[tr_idx, m_idx].spines['top'].set_visible(False)
            ax[tr_idx, m_idx].spines['right'].set_visible(False)
            plt.sca(ax[tr_idx,m_idx])
            plt.tick_params(left=False,  labelleft=False,labelbottom=False, bottom=False)
            if m_idx ==0:
                plt.ylabel(trials[tr_idx])
            tr_idx+=1
        else:
            print('mouse '+m+' does not have trial '+tr+', skipping...')
            ax[tr_idx, m_idx].spines['left'].set_visible(False)
            ax[tr_idx, m_idx].spines['bottom'].set_visible(False)
            ax[tr_idx, m_idx].spines['top'].set_visible(False)
            ax[tr_idx, m_idx].spines['right'].set_visible(False)
            plt.sca(ax[tr_idx, m_idx])
            plt.tick_params(left=False, labelleft=False, labelbottom=False, bottom=False)
            tr_idx+=1
            continue
    m_idx+=1
cbar = ax[0,0].collections[0].colorbar
cbar.set_ticklabels(behaviors)
cbar_ax.xaxis.set_ticks_position('top')
plt.subplots_adjust(left=0.01, right=.99, top=.92, bottom=0.05)
plt.savefig(output_folder+'ethogams_matrix.png',dpi=300)
plt.close()