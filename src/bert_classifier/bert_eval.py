#!/bin/env python3
# coding: utf-8

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from torchtext.data import Field, Example, Dataset, Iterator
from scipy.special import softmax



#  This scripts load your pre-trained model and your vectorizer object
#  (essentialy, vocabulary) and evaluates it on a test set.


def evaluate(model, iter, device):
    correct, total = 0, 0
    Y = np.empty(0, dtype=int)
    Y_pred = np.empty(0, dtype=int)

    print(len(iter))
    with torch.no_grad():
        for batch in tqdm(iter):
            text = batch.text
            gold = torch.LongTensor(batch.label)
            X = torch.LongTensor(text)
            X.to(device)
            gold.to(device)

            y_pred = model(X)[0]

            pred = y_pred.argmax()

            #
            # To shift the the probability from 0.5 - 0.01 in favour of 1
            #proba = softmax(y_pred)
            #proba = proba.squeeze()
            #if proba[1] > 0.01:
            #    pred = 1
            #else:
            #    pred = 0


            if pred == gold:
                correct += 1
            total += 1
            Y = np.append(Y, gold)
            Y_pred = np.append(Y_pred, pred)

    report = classification_report(Y, Y_pred)
    p_mak, r_mak, f_mak, _ = precision_recall_fscore_support(Y,
                                                             Y_pred,
                                                             average='macro')
    acc = accuracy_score(Y, Y_pred)
    return acc, report, p_mak, r_mak, f_mak


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',
                        help="PyTorch model saved with model.save()",
                        action='store',
                        default="bert_large_majority170221.bin")
                        #default="bert_large.bin")
    parser.add_argument('--test',
                        help="Test dataset)",
                        action='store',
                        #default="/home/ole/src/scope_detection/data/gold_data/all_samples_gold.tsv")
                        #default="/home/ole/src/scope_detection/data/gold_data/DNVGL-OS-E101_samples.tsv")
                        #default = "/home/ole/src/scope_detection/data/gold_data/DNVGL-RU-FD.tsv")
                        #default="/home/ole/src/scope_detection/data/gold_data/equinor_samples.tsv")

                        #default = "/home/ole/src/scope_detection/data/old/equinor_samples_gold.tsv")
                        #default="/home/ole/src/scope_detection/data/old/dnvgl-os-e101_samples_gold.tsv")
                        default="/home/ole/src/scope_detection/src/combine_samples/majority_gold.tsv")
    args = parser.parse_args()

    print("Using the following parameters:\n ")
    for arg in vars(args):
        print(arg.upper() + ": ", getattr(args, arg))
    print("\n")

    # 1) load the model
    print('Loading the model...')
    model = torch.load(args.model, map_location=torch.device('cpu'))  # Loading the model itself
    #print(model)

    # 2) Load testdata
    print('Loading the test set...')
    df = pd.read_csv(args.test, sep='\t', header=0)
    # removing ['num', 'req_id'] for classification
    df = df[['sent', 'label']]#[:200]

    BERT_MODEL = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir="./cache")
    MAX_SEQ_LEN = 32  # This number should match training
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
    fields = [('text', text_field), ('label', label_field)]

    dataset = []
    for n, entry in df.iterrows():
        # print(entry)
        example = Example.fromlist(entry, fields)
        dataset.append(example)
    dataset = Dataset(dataset, fields)
    val_iter = Iterator(dataset, batch_size=1, shuffle=False)
    print('Finished loading the test set')

    print('===========================')
    print('Evaluation:')
    print('===========================')

    device = torch.device("cpu")
    acc, report, p_mak, r_mak, f_mak = evaluate(model, val_iter, device)
    print('Classification report:')
    print(report)