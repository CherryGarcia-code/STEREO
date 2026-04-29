from glob import glob
import os
import json
import pickle
import pandas as pd


def load_csv_data(filepath):
    return pd.read_csv(filepath)

def load_json_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)
    
def save_as_pickle(file_path, data):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def load_pickle(file_path):
    with open(file_path, "rb") as file:
        preprocessed_data  =  pickle.load(file)
    return preprocessed_data

def get_cell_count_files(mouse_dir):
    cell_count_files = glob(os.path.join(mouse_dir, '*.csv'))
    # trials_files = glob(os.path.join(mouse_dir, '*__trials.json'))
    print('number of files: cell counts: ', len(cell_count_files))
    print(cell_count_files)
    return cell_count_files

def extractSectionMetadata(base_filename):
    # Split the base file name by underscore
    parts = base_filename.split('_')
    # The experimental group is the second element, with the trailing number removed
    subject_id = parts[0]
    exp_group = ''.join(filter(str.isalpha, parts[1]))

    exp_group = 'coc_20mgKg' if subject_id in ('wt2', 'c577m3') else 'coc_30mgKg' if exp_group == 'test' else exp_group

    section_number = parts[2].replace('.csv', '')
    # Return the extracted metadata
    return subject_id, exp_group, section_number


def process_section_data(cell_count_file):
    # Example for one section
    print('******************   NEW   section   ******************')

    section_data = load_csv_data(cell_count_file[0])
    
   
    # Extract the base file name without the directory
    base_filename = os.path.basename(cell_count_file[0])

    
    
    subject_id, exp_group, section_number = extractSectionMetadata(base_filename)
    section_data.drop('ImageNumber', axis=1, inplace=True)
    section_data.columns = ['cell_number', 'fos_count', 'fos_int_intensity', 'fos_mean_intensity', 'fos_median_intensity', 'x_location', 'y_location']
    section_data.insert(0, 'subject_id', subject_id)
    section_data.insert(1, 'exp_group', exp_group)
    section_data.insert(2, 'section_number', section_number)
    
    print('mouse_id is: ',subject_id)
    print('experimental group is: ', exp_group)
    print('section number is: ', section_number)
   
 
    return section_data


def flatten_nested_df(all_data_df):

    # Step 1: Expand the nested DataFrame structure into a list of DataFrames
    dataframes_list = []
    for subject_id, series in all_data_df.items():
        for section, section_df in series.items():
            if section_df is not None:
                # Assign new index levels to the session_df for proper concatenation
                # session_df = session_df.assign(subject_id=subject_id, session_id=session_id)
                dataframes_list.append(section_df)

    # Step 2: Concatenate these DataFrames into a single DataFrame with a MultiIndex
    concat_df = pd.concat(dataframes_list)
    return concat_df



