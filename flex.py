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

folder = 'Data/'
behaviors = ['Grooming', 'Body licking', 'Wall licking', 'Floor licking', 'Rearing', 'Back to camera', 'Other', 'Immobile', 'Jump']
colors = ['#c21296', '#06d6a0', '#ee476f', '#1189b1', '#ffd169', '#783f04', '#596163', '#2a363b', '#b4a7d6']
FPS=15
sample_rate=15
#%% Loading
ifile = bz2.BZ2File(folder + 'CTM.pkl', 'rb')
CTM = pickle.load(ifile)
ifile.close()
ifile = bz2.BZ2File(folder + 'CMT.pkl', 'rb')
CMT = pickle.load(ifile)
ifile.close()