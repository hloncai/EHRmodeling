#!/home/ruifengl/anaconda3/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import inflect
from tqdm import tqdm
from pytorch_pretrained_bert.tokenization import BertTokenizer
from params import *
import argparse
import os
from os import listdir
from os.path import isfile, join
import itertools
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='preprocessing help')
    parser.add_argument('-dd', '--data_dir', type=str, default='/data/users/linh/USF_Practicum/glioma_others', help='data dir')
    parser.add_argument('-t', '--task', type=str, default='breast', help='data dir')
    parser.add_argument('-o', '--output_dir', type=str, default='./output_dir', help='output dir')
    parser.add_argument('-m', '--mode', type=str, help='Run all experiments or not')
    parser.add_argument('-d', '--days', nargs='+', help='List of days', default=['180'])
    parser.add_argument('-i', '--ids', nargs='+', help='List of experiments', default=['0'])
    parser.add_argument('-dn', '--day_number', type=str, default=365, help='Number of days for experiments')
    parser.add_argument('-en', '--experiment_number', type=str, default=0, help='Experiment Number')
    return parser.parse_args()

def valid_file_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_all_files(input_dir):
    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    number_days = set()
    experiment_numbers = set()
    for f in files:
        split_file = f.split('_')
        number_days.add(int(split_file[-2]))
        experiment_numbers.add(int(split_file[-1].split('.')[0]))
        
    return sorted(number_days), sorted(experiment_numbers)

def selected_days_experiments(days, input_dir):
    days = list(map(str, days))
    experiment_numbers = defaultdict(set)
    files = [f.split('_') for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for d in days:
        for f in files:
            if f[-2] == d:
                experiment_numbers[d].add(f[-1].split('.')[0])
    return experiment_numbers

def selected_experiments_days(experiments, input_dir):
    experiments = list(map(str, experiments))
    days = defaultdict(set)
    files = [f.split('_') for f in listdir(input_dir) if isfile(join(input_dir, f))]
    for exp in experiments:
        for f in files:
            if f[-1].split('.')[0] == exp:
                days[exp].add(f[-2])
    return days
    

def reading_data(dir_path,  number_days, experiment_number, task, train = True):
    """
    read train data.
    :param dir_path: the directory of dataset
    :param number_days: the number of days for the experiment
    :param experiment_number: the number for the experiment
    :param train: the boolean value if data is training or test
    :return: return dataframe 
    """ 
    file_path = f"{dir_path}/{task}_{'train' if train else 'test'}_{number_days}_{experiment_number}.pkl"
    data = pd.read_pickle(file_path)
    return data


def text_replace(x):
    y=re.sub('\*', '', x)
    y=re.sub('\/\/', '', y)
    y=re.sub('\\\\', '', y)
    y=re.sub('[0-9]+\.  ','',y) 
    #remove 1.  , 2.   since the segmenter segments based on this. preserve 1.2 
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('admission date:','',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('--|__|==','',y)
    y=re.sub(r"\b\d+\b", lambda m: inflect.engine().number_to_words(m.group()), y) 
    # '\b \b' means whole word only
    return y

def text_clean(df): 
    df_cleaned = df.copy()
    df_cleaned['text']=df_cleaned['text'].fillna(' ')
    df_cleaned['text']=df_cleaned['text'].str.replace('\n',' ')
    df_cleaned['text']=df_cleaned['text'].str.replace('\r',' ')
    df_cleaned['text']=df_cleaned['text'].apply(str.strip)
    df_cleaned['text']=df_cleaned['text'].str.lower()

    df_cleaned['text']=df_cleaned['text'].apply(lambda x: text_replace(x))
    return df_cleaned

def truncate(df, trunk_len, overlap_len=0):
    want = pd.DataFrame({'ID':[], 'Token_trunc':[]})
    for i in range(len(df)):
        length = df['len'][i]
        n = int(np.ceil(length/(trunk_len-overlap_len)))
        for j in range(n):
            tok = df['Token'][i][j*(trunk_len-overlap_len): 
                                 j*(trunk_len-overlap_len)+trunk_len]
            want = want.append({
                'Token_trunc': tok,
                'ID': df['ptId'][i]}, ignore_index=True)
    return want


def preprocessing(input_dir, output_dir, number_days, experiment_number, task, train=True):
    data = reading_data(input_dir, number_days, experiment_number, task, train)
    df = pd.DataFrame(data)
    df['ptId'] = df.index
    df_cleaned = text_clean(df)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    df_cleaned['Token'] = df_cleaned['text'].apply(lambda x: tokenizer.tokenize(x))
    df_cleaned['len'] = df_cleaned['Token'].apply(lambda x: len(x))
    df_trunked = truncate(df_cleaned, trunk_len, overlap_len) 
    # no label needed
    output_path = f"{output_dir}/{task}_{'train' if train else 'test'}_{number_days}_{experiment_number}_tokens.pkl"
    print(output_path)
    df_trunked.to_pickle(output_path)

def main():
    args = parse_args()
    valid_file_path(args.output_dir)
    if args.mode == 'all':
        print(f'mode: {args.mode}')
        number_days, experiment_numbers = get_all_files(args.data_dir)
        for day, experiment in itertools.product(number_days, experiment_numbers):
            preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, True)
            preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, False)
    elif args.mode == 'days':
        print(f'mode: {args.mode}')
        # run all the experiment with the day regardless of the id
        experiment_numbers = selected_days_experiments(args.days, args.data_dir)
        for day, experiments in experiment_numbers.items():
            for experiment in experiments:
                print(day, experiment)
                preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, True)
                preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, False)
    elif args.mode == 'ids':
        print(f'mode: {args.mode}')
        # run all the experiment with the id regardless of the day
        days = selected_experiments_days(args.ids, args.data_dir)
        for experiment, days in days.items():
            for day in days:
                print(day, experiment)
                preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, True)
                preprocessing(args.data_dir, args.output_dir, day, experiment, args.task, False)
        
    else:
        print(args.day_number, args.experiment_number)
        preprocessing(args.data_dir, args.output_dir, args.day_number, args.experiment_number, args.task, True)
        preprocessing(args.data_dir, args.output_dir, args.day_number, args.experiment_number, args.task, False)
    
    
    
if __name__ == '__main__':
    main()
    

    
# add arguements so user can give inputs 
# one specific experiment or all experments

# preprcoessing -mode all|one -days 30 -experiment_id 5

# model 

# train valid ,test on the specific experimetn or all

### try git clone of subdirectory 