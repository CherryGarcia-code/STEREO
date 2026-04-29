import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import os
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import medfilt, butter, filtfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize
import bz2
import pickle
# The following pipeline is based on Simpson et al. 2024 approach
cohort = 'a2a_hm3dq'
mouse = 'cA184m7'
output_folder = 'output/Photometry/'+cohort+'/'+mouse
if not os.path.exists(output_folder):os.makedirs(output_folder)
base_dir = 'Z:/CNN/DREADDS/updated_files/'+cohort+'/'+mouse+'/videos/'
csv_file = [f for f in os.listdir(base_dir) if f.endswith('csv') and 'cocaine5' in f][0]

csv_path = base_dir+csv_file
data = pd.DataFrame(pd.read_csv(csv_path))
if mouse.startswith('cA242'):
    iso_left = np.array(data.loc[data['Flags'] == 17, 'Region3G'])
    iso_right = np.array(data.loc[data['Flags'] == 17, 'Region2G'])
    sig_left = np.array(data.loc[data['Flags'] == 18, 'Region3G'])
    sig_right = np.array(data.loc[data['Flags'] == 18, 'Region2G'])
    timestamps = np.array(data.loc[data['Flags'] == 18, 'Timestamp'])
else:
    iso_left = np.array(data.loc[data['LedState'] == 1, 'Region3G'])
    iso_right = np.array(data.loc[data['LedState'] == 1, 'Region2G'])
    sig_left = np.array(data.loc[data['LedState'] == 2, 'Region3G'])
    sig_right = np.array(data.loc[data['LedState'] == 2, 'Region2G'])
    timestamps = np.array(data.loc[data['LedState'] == 2, 'Timestamp'])

sr = np.mean(1/np.diff(timestamps))
number_of_samples = np.amin([iso_left.shape[0], sig_left.shape[0]])
iso_left = iso_left[:number_of_samples]
iso_right = iso_right[:number_of_samples]
sig_left = sig_left[:number_of_samples]
sig_right = sig_right[:number_of_samples]
timestamps = timestamps[:number_of_samples]



#%% Plotting raw signal
fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].plot(iso_left)
ax[0,0].set_title('Isobestic left')
ax[1,0].plot(sig_left,c='green')
ax[1,0].set_title('GCaMP left')
ax[0,1].plot(iso_right)
ax[0,1].set_title('Isobestic right')
ax[1,1].plot(sig_right,c='green')
ax[1,1].set_title('GCaMP right')
plt.suptitle('Raw traces')
plt.tight_layout()
plt.savefig(output_folder+'rawtraces.png',dpi=300)
plt.savefig(output_folder+'rawtraces.pdf',dpi=300)
#%% De-noising the signal
b,a = butter(2,10,btype='low',fs = sr)
iso_right_denoised = filtfilt(b,a,iso_right)
iso_left_denoised = filtfilt(b,a,iso_left)
sig_right_denoised = filtfilt(b,a,sig_right)
sig_left_denoised = filtfilt(b,a,sig_left)
fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].plot(iso_left_denoised,alpha=.3,label='De-noised')
ax[0,0].plot(iso_left,label = 'Raw',alpha=.3)
ax[0,0].set_title('Isobestic left')
ax[0,0].legend()
ax[1,0].plot(sig_left_denoised,alpha=.3,label='De-noised')
ax[1,0].plot(sig_left,alpha=.3,label='Raw',c='green')
ax[1,0].set_title('GCaMP left')
ax[1,0].legend()
ax[0,1].plot(iso_right_denoised,label='De-noised',alpha=.3)
ax[0,1].plot(iso_right,label='Raw',alpha=.3)
ax[0,1].set_title('Isobestic right')
ax[0,1].legend()
ax[1,1].plot(sig_right_denoised,label='De-noised',alpha=.3)
ax[1,1].plot(sig_right,label='Raw',alpha=.3,c='green')
ax[1,1].set_title('GCaMP right')
ax[1,1].legend()
plt.suptitle('De-noising')
plt.tight_layout()
plt.savefig(output_folder+'denoised_traces.png',dpi=300)
plt.savefig(output_folder+'denoised_traces.pdf',dpi=300)
#%% Fitting a curve for the photobleaching decay
bleaching_cutoff_freq=0.005
b,a = butter(2,bleaching_cutoff_freq,btype='low',fs = sr)
sig_right_decfit = filtfilt(b,a,sig_right_denoised)
iso_right_decfit = filtfilt(b,a,iso_right_denoised)
sig_left_decfit = filtfilt(b,a,sig_left_denoised)
iso_left_decfit = filtfilt(b,a,iso_left_denoised)
fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].plot(timestamps,iso_left_denoised,label = 'Iso left',alpha=.5)
ax[0,0].plot(timestamps,iso_left_decfit,label = 'Iso left decay fit')
ax[0,0].legend()
ax[0,0].set_title('Isobestic left')
ax[1,0].plot(sig_left_denoised,label = 'GCaMP right',alpha=.5,c='green')
ax[1,0].plot(sig_left_decfit,label = 'GCaMP right decay fit')
ax[1,0].legend()
ax[1,0].set_title('GCaMP left')
ax[0,1].plot(iso_right_denoised,label='Iso right',alpha=.5)
ax[0,1].plot(iso_right_decfit,label='Iso right decay fit')
ax[0,1].legend()
ax[0,1].set_title('Isobestic right')
ax[1,1].plot(sig_right_denoised,label = 'GCaMP right',alpha=.5,c='green')
ax[1,1].plot(sig_right_decfit,label='GCaMP right decay fit')
ax[1,1].legend()
ax[1,1].set_title('GCaMP right denoised')
plt.suptitle('Bleaching fit (cutoff freq = '+str(bleaching_cutoff_freq)+'Hz)')
plt.tight_layout()
plt.savefig(output_folder+'decay_fit.png',dpi=300)
plt.savefig(output_folder+'decay_fit.pdf',dpi=300)
#%% Removing the decay
sig_right_noBleaching = sig_right_denoised-sig_right_decfit
iso_right_noBleaching = iso_right_denoised-iso_right_decfit
sig_left_noBleaching = sig_left_denoised-sig_left_decfit
iso_left_noBleaching = iso_left_denoised-iso_left_decfit
fig,ax = plt.subplots(nrows=2,ncols=2)
ax[0,0].plot(iso_left_noBleaching)
ax[0,0].set_title('Isobestic left')
ax[1,0].plot(sig_left_noBleaching,c='green')
ax[1,0].set_title('GCaMP left')
ax[0,1].plot(iso_right_noBleaching)
ax[0,1].set_title('Isobestic right')
ax[1,1].plot(sig_right_noBleaching,c='green')
ax[1,1].set_title('GCaMP right')
plt.suptitle('Photobleaching correction')
plt.tight_layout()
plt.savefig(output_folder+'decay_removal.png',dpi=300)
plt.savefig(output_folder+'decay_removal.pdf',dpi=300)
#%%Calculation motion artifact by regressing isobsetic channel against its signal channel
slope_right, intercept_right, r_value_right, p_value_right, std_err_right = linregress(x=iso_right_noBleaching, y=sig_right_noBleaching)
slope_left, intercept_left, r_value_left, p_value_left, std_err_left = linregress(x=iso_left_noBleaching, y=sig_left_noBleaching)
fig,ax=plt.subplots(ncols=2)
plt.sca(ax[0])
plt.scatter(iso_left_noBleaching[::5],sig_left_noBleaching[::5],alpha=.1,marker='.')
plt.xlabel('Isobestic')
plt.ylabel('GCaMP')

x = np.array(plt.xlim())
plt.plot(x, intercept_left+slope_left*x)
plt.title('Left hemisphere\nSlope: {:.3f}'.format(slope_left)+' R-squared: {:.3f}'.format(r_value_left**2))
plt.sca(ax[1])
plt.scatter(iso_right_noBleaching[::5],sig_right_noBleaching[::5],alpha=.1,marker='.')
plt.xlabel('Isobestic')
plt.ylabel('GCaMP')
x = np.array(plt.xlim())
plt.plot(x, intercept_right+slope_right*x)
plt.title('Right hemisphere\nSlope: {:.3f}'.format(slope_right)+' R-squared: {:.3f}'.format(r_value_right**2))
plt.suptitle('Isobestic and signal linear fit')
sig_right_est_motion = intercept_right+slope_right*iso_right_noBleaching
sig_right_motion_corrected = sig_right_noBleaching-sig_right_est_motion
plt.tight_layout()
plt.savefig(output_folder+'iso_sig_regression.png',dpi=300)
plt.savefig(output_folder+'iso_sig_regression.pdf',dpi=300)
#%% Estimating motion artifact from a scaled version of the isobestic channel
sig_left_est_motion = intercept_left+slope_left*iso_left_noBleaching
sig_left_motion_corrected = sig_left_noBleaching-sig_left_est_motion

fig,ax = plt.subplots(ncols=2)
ax[0].plot(sig_left_noBleaching,alpha=.3,label='before')
ax[0].plot(sig_left_motion_corrected,alpha=.3,label='after')
ax[0].plot(sig_left_est_motion,alpha=.3,label='motion')
ax[0].legend()
ax[0].set_title('Left hemisphere')
ax[1].plot(sig_right_noBleaching,alpha=.3,label='before')
ax[1].plot(sig_right_motion_corrected,alpha=.3,label='after')
ax[1].plot(sig_right_est_motion,alpha=.3,label='motion')
ax[1].legend()
ax[1].set_title('Right hemisphere')
plt.suptitle('Motion correction')
plt.tight_layout()
plt.savefig(output_folder+'motion_correceted_sig.png',dpi=300)
plt.savefig(output_folder+'motion_correceted_sig.pdf',dpi=300)
#%% Calculating df/f
right_df_f = 100*sig_right_motion_corrected/sig_right_decfit
left_df_f = 100*sig_left_motion_corrected/sig_left_decfit
fig,ax = plt.subplots(ncols=2)
ax[0].plot(left_df_f,c='green')
ax[0].set_ylabel('$\Delta$F/F (%)')
ax[0].set_xlabel('Time (frames)')
ax[0].set_title('Left hemisphere')
ax[1].plot(right_df_f,c='green')
ax[1].set_ylabel('$\Delta$F/F (%)')
ax[1].set_xlabel('Time (frames)')
ax[1].set_title('Right hemisphere')
plt.suptitle('$\Delta$F/F')
plt.tight_layout()
plt.savefig(output_folder+'df_over_f.png',dpi=300)
plt.savefig(output_folder+'df_over_f.pdf',dpi=300)
final_sig = {'left':left_df_f , 'right':right_df_f}
# TODO: add reasmpling?
ofile = bz2.BZ2File(output_folder+'final_signal.pkl', 'wb')
pickle.dump(final_sig, ofile)
ofile.close()
