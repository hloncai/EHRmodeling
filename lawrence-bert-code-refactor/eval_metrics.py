import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def compute_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return accuracy, precision, f1, recall


def convert_to_csv(results, day, output_dir):
    # tranform keys
    results = {f'Experiment {k}': v for k, v in results.items()}
    # convert to df
    results_df = pd.DataFrame.from_dict(results, orient='index',
                       columns=['AUC', 'Precision', 'F1', 'Recall'])
    # calculate avg and std
    avg = [results_df['AUC'].mean(), results_df['Precision'].mean(),results_df['F1'].mean(),results_df['Recall'].mean()]
    std = [results_df['AUC'].std(), results_df['Precision'].std(),results_df['F1'].std(),results_df['Recall'].std()]
    
    # calculate +/- std
    minus_std = (np.array(avg)-np.array(std)).tolist()
    plus_std = (np.array(avg)+np.array(std)).tolist()
    
    # append statistics
    statistic_df = pd.DataFrame([avg, std, minus_std, plus_std], columns=['AUC', 'Precision', 'F1', 'Recall'], index=['Average', 'Standard Deviation', 'minus_std', 'plus_std'])
    
    # concat statistics with df
    results_df = results_df.append(statistic_df)
    
    # convert to csv
    today = date.today().strftime('%m-%d-%Y')
    results_df.to_csv(f'./{output_dir}/results-day-{day}-{today}.csv')
    # return results_df