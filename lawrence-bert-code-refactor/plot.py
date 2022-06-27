from modeling import *
import argparse
from eval_metrics import * 
import matplotlib.pyplot as plt
import statistics
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc, confusion_matrix, classification_report

def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default="/data/users/linh/USF_Practicum/glioma_tokenized",
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--raw_data_dir",
                        default="/data/users/linh/USF_Practicum/glioma_other",
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task",
                        default="glioma",
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--num_day",
                        default=180,
                        type=int,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--experiment_number",
                        default=21,
                        type=int,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--tokenizer",
                        default='bert-base-uncased',
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model",
                        default='./pretraining',
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--out_directory",
                        default='./results',
                        type=str,
                        # required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    return parser.parse_args()

def metrics_func(roc_auc, f1, accuracy, precision, recall):
    mean_auc = round(statistics.mean(roc_auc), 3) 
    std_auc = round(statistics.stdev(roc_auc), 3)
    mean_accuracy = round(statistics.mean(accuracy), 3) 
    std_accuracy = round(statistics.stdev(accuracy), 3)
    mean_precision = round(statistics.mean(precision), 3) 
    std_precision = round(statistics.stdev(precision), 3)
    mean_f1 = round(statistics.mean(f1), 3) 
    std_f1 = round(statistics.stdev(f1), 3)
    mean_recall = round(statistics.mean(recall), 3) 
    std_recall = round(statistics.stdev(recall), 3)

    return mean_auc, std_auc, mean_accuracy, std_accuracy, mean_precision, std_precision, mean_f1, std_f1, mean_recall, std_recall

def plot_viz(tprs, base_fpr, metric_results, num_day, task, out_directory):
    fig, ax = plt.subplots()
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std

    plt.plot(base_fpr, mean_tprs, 'b')
    plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    # plt.show()
    mean_auc, std_auc, mean_accuracy, std_accuracy, mean_precision, std_precision, mean_f1, std_f1, mean_recall, std_recall = metric_results
    textstr = '\n'.join((
    f'AUROC: , {mean_auc} +/- {std_auc}',
    f'Accuracy: , {mean_accuracy} +/- {std_accuracy}',
    f'Precision: , {mean_precision} +/- {std_precision}',
    f'F1: , {mean_f1} +/- {std_f1}',
    f'Recall: , {mean_recall} +/- {std_recall}'))
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.975, 0.04, textstr, transform=ax.transAxes, fontsize=10,
            va='bottom', ha='right', bbox=props)
    
    plt.savefig(f'{out_directory}/{task}_{num_day}.png')
    

def main():
    args = parse_args()
    tprs = []
    roc_auc, f1, accuracy, precision, recall = ([] for i in range(5))
    for num in range(args.experiment_number):
        print(f'num_days: {args.num_day}, experiment_number: {num}')
        df_train = loading_data(args.data_dir,  args.num_day, num, args.task, True)
        df_test = loading_data(args.data_dir,  args.num_day, num, args.task, False)

        # model
        tokenizer = get_tokenizer(args.tokenizer)
        model = BertForSequenceClassification.from_pretrained(args.bert_model, 1)

        # train data
        train_dataloader = transform_data(df_train, tokenizer, args.max_seq_length, args.train_batch_size)
        train_outputs = generate_outputs(train_dataloader, model)
        train_array_grouped_dict = max_pooling_id(df_train, train_outputs)
        train_data_agg = data_vector(train_array_grouped_dict, args.raw_data_dir,  args.num_day, num, args.task, True)

        # test data
        test_dataloader = transform_data(df_test, tokenizer, args.max_seq_length, args.eval_batch_size)
        test_outputs = generate_outputs(test_dataloader, model)
        test_array_grouped_dict = max_pooling_id(df_test, test_outputs)
        test_data_agg = data_vector(test_array_grouped_dict, args.raw_data_dir,  args.num_day, num, args.task, False)

        # logistic regression
        y_test, y_pred, y_pred_proba = logit_reg(train_data_agg, test_data_agg)

        # viz
        base_fpr = np.linspace(0, 1, 101)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
        roc_auc.append(auc(fpr, tpr))
        accuracy.append(accuracy_score(y_test, y_pred) )
        precision.append(precision_score(y_test, y_pred) )
        f1.append(f1_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred) )
        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    metric_results = metrics_func(roc_auc, f1, accuracy, precision, recall)
    plot_viz(tprs, base_fpr, metric_results, args.num_day, args.task, args.out_directory)
    
    
# def main():
#     args = parse_args()
#     modeling(args.data_dir,  raw_data_dir, number_days, experiment_number)

    
    
    
if __name__ == '__main__':
    main()