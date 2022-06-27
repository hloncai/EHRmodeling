#!/home/ruifengl/anaconda3/bin/python3
import pandas as pd
import logging
from pytorch_pretrained_bert.tokenization import BertTokenizer
from params import *
from tqdm import trange, tqdm
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
import argparse

import torch
from torch.utils.data import TensorDataset, DataLoader

import py_compile
py_compile.compile('modeling_readmission.py')

from modeling_readmission import BertGetCLSOutput, BertForSequenceClassification

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def _cuda():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def loading_data(input_dir,  number_days, experiment_number, task, train = True):
    data_path = f"{input_dir}/{task}_{'train' if train else 'test'}_{number_days}_{experiment_number}_tokens.pkl"
    data = pd.read_pickle(data_path)
    data['ID'] = data['ID'].apply(lambda x: int(x))
    return data
    

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        
def convert_examples_to_features(data, max_seq_length, tokenizer):
    features = []
    for i in range(len(data)):
        tokens_import = data['Token_trunc'][i]
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_import:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens) 
        # BertTokenizer.from_pretrained('bert-base-uncased').convert_tokens_to_ids
        # convert tokens to number (max 512)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if i < 5:
            logger.info("*** Example ***")
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


    
    
def transform_data(data, tokenizer, max_seq_length, batch_size):
    features = convert_examples_to_features(data, max_seq_length, tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    ds = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    dl = DataLoader(ds, #sampler=eval_sampler, 
                                  batch_size=batch_size)
    return dl

def generate_outputs(dataloader, model):
    device = _cuda()

    model.to(device)
    model.eval()
    outputs = []
    for input_ids, input_mask, segment_ids in tqdm(dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            _, cls_output_t = model(input_ids, segment_ids, input_mask)
            cls_output_t = cls_output_t.cpu().numpy()
            for v in cls_output_t:
                outputs.append(v)
    return outputs

def max_pooling_id(data, outputs):
    keys = np.array(data['ID'])
    vals = np.asarray(outputs)
    array_grouped_dict = {key: vals[keys == key] for key in np.unique(keys)}
    for key in array_grouped_dict:
        array_grouped_dict[key] = array_grouped_dict[key].max(axis=0)
    return array_grouped_dict

def loading_raw(dir_path,  number_days, experiment_number, task, train = True):
    file_path = f"{dir_path}/{task}_{'train' if train else 'test'}_{number_days}_{experiment_number}.pkl"
    data = pd.read_pickle(file_path)
    return data

def data_vector(array_grouped_dict, raw_data_dir,  number_days, experiment_number, task, train = True):
    data = loading_raw(raw_data_dir,  number_days, experiment_number, task, train = train)
    data['ptId'] = data.index
    data_agg = pd.DataFrame(data)
    data_agg['vector'] = data_agg['ptId'].map(array_grouped_dict)
    return data_agg

def logit_reg(train_data, test_data):
    X = torch.tensor(list(train_data['vector'].values) )
    y = torch.tensor(list(train_data['label'].values)).reshape(-1,1).float()
    X_test = torch.tensor(list(test_data['vector'].values) )
    y_test = torch.tensor(list(test_data['label'].values)).reshape(-1,1).float()
    logit = LogisticRegression(C=5e1, solver='lbfgs', random_state=17, n_jobs=4) # grid search 
    logit.fit(X, y)
    y_pred = logit.predict(X_test)
    y_pred_proba = logit.predict_proba(X_test)[:,1]
    return y_test, y_pred, y_pred_proba

def get_tokenizer(tokenizer):
    return BertTokenizer.from_pretrained(tokenizer)

    
def modeling(input_dir, raw_data_dir, number_days, experiment_number):
    
    df_train = loading_data(input_dir,  number_days, experiment_number, task, True)
    df_test = loading_data(input_dir,  number_days, experiment_number, task, False)
    
    # model
    tokenizer = get_tokenizer('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(pretrain_bert_model, 1)
    
    # train data
    train_dataloader = transform_data(df_train, tokenizer, max_seq_length, batch_size)
    train_outputs = generate_outputs(train_dataloader, model)
    train_array_grouped_dict = max_pooling_id(df_train, train_outputs)
    train_data_agg = data_vector(train_array_grouped_dict, raw_data_dir,  number_days, experiment_number, task, True)
    
    # test data
    test_dataloader = transform_data(df_test, tokenizer, max_seq_length, batch_size)
    test_outputs = generate_outputs(test_dataloader, model)
    test_array_grouped_dict = max_pooling_id(df_test, test_outputs)
    test_data_agg = data_vector(test_array_grouped_dict, raw_data_dir,  number_days, experiment_number, task, False)
    
    # logistic regression
    y_test, y_pred, y_pred_proba = logit_reg(train_data_agg, test_data_agg)
    
    # viz
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba, pos_label=1)
    plt.plot(fpr, tpr, 'b', alpha=0.15)
    tpr = np.interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)
    
    return tprs
    
    
# def main():
#     args = parse_args()
#     modeling(args.data_dir,  raw_data_dir, number_days, experiment_number)

    
    
    
# if __name__ == '__main__':
#     main()