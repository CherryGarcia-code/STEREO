import numpy as np
import pandas as pd
import os

data_dir = 'Z:/CNN/DREADDS/updated_files/'
for cohort in ['drd1_hm4di','drd1_hm3dq','controls','a2a_hm4di','a2a_hm3dq']:
    for mouse in os.listdir(data_dir+cohort):
        mouse_folder = data_dir+cohort+'/'+mouse+'/videos/'
        cs_file = open(mouse_folder + 'codingScheme.txt', "r")
        if  cs_file.readline()!='1+1+1': continue
        else:
            for file in os.listdir(mouse_folder):
                if file.endswith('csv'):
                    data = pd.DataFrame(pd.read_csv(mouse_folder+file))
                    if mouse.startswith('cA242') or mouse.startswith('cg') or mouse in ['cA265m2', 'cA265m3', 'cA265m4', 'cA265m5']:
                        timestamps = np.array(data.loc[data['LedState'] == 2, 'Timestamp'])

                    else:
                        timestamps = np.array(data.loc[data['Flags'] == 18, 'Timestamp'])
                    sample_rate = np.mean(1/np.diff(timestamps))
                    # if sample_rate>30:
                    print(cohort,mouse,file,sample_rate)


