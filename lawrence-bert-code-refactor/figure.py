import pandas as pd
import numpy as np
import argparse
import statistics
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt

'''
! python ./figure.py \
    --file_paths ./pred_outputs/breast ./pred_outputs/glioma ./pred_outputs/lung ./pred_outputs/prostate \
    --experiment_nums 21 \
    --save_path ./results/plot_figure.png
'''


def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--file_paths",
                        nargs='+', 
                        default=[],
                        type=str,
                        # required=True,
                        help="The list of file path that point to breast, glioma, lung and prostate directories")
    parser.add_argument("--experiment_nums",
                        default=21,
                        type=int,
                        # required=True,
                        help="The total number of experiment for each day.")
    parser.add_argument("--save_path",
                        default=None,
                        type=str,
                        # required=True,
                        help="The directory path that saves plots")
    return parser.parse_args()

# loading a pickle object 
def loading_pickle(filepath):
    data = pd.read_pickle(filepath)  
    return data

# calculate fpr, tpr and accuracy given predicted results from bert and logistic regression
def metrics_calc(filepath, task, experiment_nums):
    tprs = []
    acc = []
    for i in range(experiment_nums):
        result = loading_pickle(filepath + f'/{task}_180_{i}.pkl')
        y_test = result['y_test']
        y_pred_proba = result['y_pred_proba']
        
        base_fpr = np.linspace(0, 1, 101)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
        # acc.append(accuracy_score(y_test, y_pred))
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    # return tprs, round(statistics.mean(accuracy), 3), round(statistics.stdev(accuracy), 3)
    return tprs

# making plot with boundries
def plot_func(tprs_all, output_dir):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    base_fpr = np.linspace(0, 1, 101)
    keys = list(tprs_all.keys())
    
    # breast
    tprs = np.array(tprs_all[keys[0]])
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    axs[0, 0].plot(base_fpr, mean_tprs, 'b')
    axs[0, 0].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    # placeholder for tfidf 
    
    axs[0, 0].plot([0, 1], [0, 1],'b--', alpha=0.3)
    axs[0, 0].axis(xmin=-0.01,xmax=1.01, ymin=-0.01,ymax=1.01)
    axs[0, 0].set_ylabel('True Positive Rate')
    axs[0, 0].set_xlabel('False Positive Rate')
    axs[0, 0].set_title(keys[0])


    # glioma
    tprs = np.array(tprs_all[keys[1]])
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    axs[0, 1].plot(base_fpr, mean_tprs, 'tab:green')
    axs[0, 1].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    # placeholder for tfidf 
    
    axs[0, 1].plot([0, 1], [0, 1],'b--', alpha=0.3)
    axs[0, 1].axis(xmin=-0.01,xmax=1.01, ymin=-0.01,ymax=1.01)
    axs[0, 1].set_ylabel('True Positive Rate')
    axs[0, 1].set_xlabel('False Positive Rate')
    axs[0, 1].set_title(keys[1])

    # lung
    tprs = np.array(tprs_all[keys[2]])
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    axs[1, 0].plot(base_fpr, mean_tprs, 'tab:red')
    axs[1, 0].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    # placeholder for tfidf 
        
    axs[1, 0].plot([0, 1], [0, 1],'b--', alpha=0.3)
    axs[1, 0].axis(xmin=-0.01,xmax=1.01, ymin=-0.01,ymax=1.01)
    axs[1, 0].set_ylabel('True Positive Rate')
    axs[1, 0].set_xlabel('False Positive Rate')
    axs[1, 0].set_title(keys[2])

    # prostate
    tprs = np.array(tprs_all[keys[3]])
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    axs[1, 1].plot(base_fpr, mean_tprs, 'tab:orange')
    axs[1, 1].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    # placeholder for tfidf 
    
    axs[1, 1].plot([0, 1], [0, 1],'b--', alpha=0.3)
    axs[1, 1].axis(xmin=-0.01,xmax=1.01, ymin=-0.01,ymax=1.01)
    axs[1, 1].set_ylabel('True Positive Rate')
    axs[1, 1].set_xlabel('False Positive Rate')
    axs[1, 1].set_title(keys[3])
    
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    fig.savefig(output_dir)
        
def main():
    args = parse_args()
    # filepaths = ['./pred_outputs/breast', './pred_outputs/glioma', './pred_outputs/lung', './pred_outputs/prostate']
    filepaths = args.file_paths
    tasks = [ i.split('/')[-1] for i in filepaths]
    experiment_nums = args.experiment_nums
    tprs = {}
    for i in range(len(filepaths)):
        tprs[tasks[i]] = metrics_calc(filepaths[i], tasks[i], experiment_nums)
    
    plot_func(tprs, args.save_path)
    
if __name__ == '__main__':
    main()