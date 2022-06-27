#!/home/ruifengl/anaconda3/bin/python3
from modeling import *
from preprocessing import *
import argparse
from eval_metrics import * 
import matplotlib.pyplot as plt


# raw_data_dir = "/data/users/linh/USF_Practicum/glioma_others"

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
    parser.add_argument("--bert_model", default='./pretraining', type=str, 
                        # required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    
    parser.add_argument("--readmission_mode", default = None, type=str, help="early notes or discharge summary")
    
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        # required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='./results',
                        type=str,
                        # required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--number_days', nargs='+', help='List of days', default=['180'])
    parser.add_argument('--experiment_number',
                        type=int, default=21,
                        help='The experiment number')
    parser.add_argument('--tokenizer',
                        type=str, default='bert-base-uncased',
                        help='The BERT tokenizer')
    
    return parser.parse_args()


    
# modeling 
if __name__ == '__main__':
    args = parse_args()
    # print(args)
    print(f'data dir: {args.data_dir}')
    print(args.number_days)
    for day in args.number_days:
        results = {}
        tprs = []
        plt.figure(figsize=(5, 5))
        plt.axes().set_aspect('equal', 'datalim')
        for num in range(args.experiment_number):
            print(f'number_days: {day}, experiment_number: {num}')
            df_train = loading_data(args.data_dir,  int(day), num)
            df_test = loading_data(args.data_dir,  int(day), num, False)

            # train data
            train_dataloader = transform_data(df_train, args.tokenizer, args.max_seq_length, args.train_batch_size)
            model = BertForSequenceClassification.from_pretrained(args.bert_model, 1)
            train_outputs = generate_outputs(train_dataloader, model)
            train_array_grouped_dict = max_pooling_id(df_train, train_outputs)
            train_data_agg = data_vector(train_array_grouped_dict, args.raw_data_dir, day, num, True)

            # test data
            test_dataloader = transform_data(df_test, args.tokenizer, args.max_seq_length, args.eval_batch_size)
            test_outputs = generate_outputs(test_dataloader, model)
            test_array_grouped_dict = max_pooling_id(df_test, test_outputs)
            test_data_agg = data_vector(train_array_grouped_dict, args.raw_data_dir, day, num, False)

            # logistic regression
            y_test, y_pred, y_pred_proba = logit_reg(train_data_agg, test_data_agg)
            
            # viz
            
            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)

            results[num]= compute_metrics(y_test, y_pred)
        convert_to_csv(results, day, args.output_dir)
