import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
import data_io
model = 'model1'
folder = 'data/'
CMT = data_io.load_CMT(flatten=True)
print('***All dictionaries were loaded***')
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Undefined', 'Locomotion','Stationary', 'Jump']
colors = data_io.palette_for(behaviors)
# This script indexes behaviours by position in the `behaviors` list above,
# which is NOT the canonical Apr26 order (note Locomotion/Stationary are also
# swapped relative to the other legacy scripts). Permute by name so a mismatch
# raises rather than silently mislabelling.
_relabel = data_io.relabel_lut(behaviors)
for _l1 in CMT.values():
    for _l2 in _l1.values():
        for _e in _l2.values():
            _e['merged'] = _relabel[_e['merged']]
FPS=15
sample_rate=15
second = 15
minute = 60*second
mpl.rcParams['xtick.labelsize']=10
mpl.rcParams['ytick.labelsize']=10
mpl.rcParams['legend.fontsize']=8
num_of_behaviors = len(behaviors)
output_folder = 'output/LocomotionBasedSeparation/Ethograms/'
SAMPLE_SIZE=10
stationary_theta = 0.021/float(FPS)
for c in CMT:
    c_output_folder = output_folder+c+'/'
    os.makedirs(c_output_folder, exist_ok=True)
    for m in CMT[c]:
        for tr in CMT[c][m]:
            print(c , m , tr)
            predictions = CMT[c][m][tr]['merged']
            predictions[predictions==7] = 8
            velocity = CMT[c][m][tr]['topcam']['velocity']
            if velocity.size!= predictions.size:
                velocity = velocity[9:]
                vigor = vigor[9:]
            stationary_indices = np.intersect1d(np.nonzero(predictions==6)[0],np.nonzero(velocity<=stationary_theta)[0])
            predictions[stationary_indices]=7
            predictions = np.reshape(predictions, (1, predictions.size))
            vigor = np.max(np.vstack([CMT[c][m][tr]['SSD']['cam1'],CMT[c][m][tr]['SSD']['cam2']]),axis=0)
            if 'photom' in CMT[c][m][tr]:
                photom = CMT[c][m][tr]['photom']
                fig, ax, = plt.subplots(4, sharex=True, figsize=(24, 8))
                ax[0] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,
                                    cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5,
                                              'drawedges': True,
                                              'use_gridspec': False, 'location': "top", 'aspect': 40}, vmin=0, vmax=8, ax=ax[0])
                ax[0].set_ylabel('Predictions', va='center')
                cbar = ax[0].collections[0].colorbar
                cbar.set_ticklabels(behaviors)
                cbar.ax.tick_params(labelsize=9)
                ax[1].plot(velocity * FPS,lw=.7,alpha=.5)
                ax[1].bar(x = stationary_indices,height =  np.max(velocity*FPS),color='#BCBDDC',width=1)
                ax[1].spines['right'].set_visible(False)
                ax[1].spines['top'].set_visible(False)
                ax[1].set_ylabel('Velocity\n(m/s)')
                ax[2].plot(vigor, alpha=.5,lw=.7)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.sca(ax[2])
                plt.ylabel('Motion index\n(a.u.)')
                # photom[hemisphere] is {'df': ..., 'z': ...}; this axis is labelled
                # Z-score, so plot the z trace rather than the dict.
                if 'left' in photom:
                    ax[3].plot(photom['left']['z'], label='Left hemisphere', alpha=.5, c='green')
                if 'right' in photom:
                    ax[3].plot(photom['right']['z'], label='Right hemisphere', alpha=.5, c='red')
                ax[3].spines['right'].set_visible(False)
                ax[3].spines['top'].set_visible(False)
                plt.sca(ax[3])
                plt.ylabel('Photometry\n(Z-score)')
                plt.legend()
            else:
                fig, ax, = plt.subplots(3, sharex=True, figsize=(24, 8))
                ax[0] = sns.heatmap(predictions, yticklabels=[''], cmap=colors,
                                    cbar_kws={'ticks': np.arange(len(behaviors)), 'boundaries': np.arange(-1, len(behaviors)) + 0.5,
                                              'drawedges': True,
                                              'use_gridspec': False, 'location': "top", 'aspect': 35}, vmin=0, vmax=8, ax=ax[0])
                ax[0].set_ylabel('Predictions', va='center')
                cbar = ax[0].collections[0].colorbar
                cbar.set_ticklabels(behaviors)
                ax[1].plot(velocity * FPS, alpha=.5, lw = .7)
                ax[1].bar(x = stationary_indices,height =  np.max(velocity*FPS),color='#BCBDDC',alpha = .3,width=1)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.sca(ax[1])
                plt.ylabel('Velocity\n(m/s)')

                ax[2].plot(vigor, alpha=.5, lw = .7)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().spines['top'].set_visible(False)
                plt.sca(ax[2])
                plt.ylabel('Motion index\n(a.u.)')
            plt.xlabel('Time (minutes)')
            plt.xticks(np.arange(0, predictions.shape[1], 900),
                       (np.arange(SAMPLE_SIZE, predictions.shape[1] + SAMPLE_SIZE, 900) / 900).astype(int), rotation=0, fontsize=8)
            plt.suptitle(m+ ' , '+tr)
            # plt.tight_layout()
            plt.savefig(c_output_folder + m+'_'+tr+'_ethogram.png', bbox_inches='tight', dpi=300)
            plt.close()


