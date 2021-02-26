#!/bin/env python3
# coding: utf-8

from argparse import ArgumentParser
import pandas as pd
import torch
from src.io_handler import read_requirements
from transformers import BertTokenizer
from tqdm import tqdm


def classify(model, df, tokenizer, file_path, device):
    with open(file_path, 'w') as F:
        F.write("sec\treq\tsent_num\tsent\tlabel\n")
        print(len(df))
        num_sents = 0
        num_scopes = 0
        with torch.no_grad():  # todo: Expecting this to fail on large sentences, as model is only 32 tokens
            for i, row in tqdm(df.iterrows(), total=len(df)):
                sent = row['sent']
                X = tokenizer.encode(sent)
                X = torch.LongTensor(X)
                X = torch.unsqueeze(X, 0)
                X.to(device)
                y_pred = model(X)[0]
                pred = y_pred.argmax()
                if pred.item() == 1:
                    num_scopes += 1
                F.write(str(row['sec']) + "\t" + str(row['req']) + "\t" + str(row['sent_num']) + "\t" + sent + "\t" + str(pred.item()) + "\n")
                num_sents += 1
    return num_sents, num_scopes


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model',
                        help="PyTorch model saved with model.save()",
                        action='store',
                        default="bert_large_majority170221.bin")
    parser.add_argument('--data',
                        help="dataset)",
                        action='store',
                        default="/home/ole/src/scope_detection/data/gold_data/all_samples_gold.tsv")
    parser.add_argument("--file",
                        help="file to write classification",
                        action='store',
                        default='/home/ole/src/scope_detection/data/other/all_samples_gold_classified.tsv')
    args = parser.parse_args()

    # 1) load the model
    print('Loading the model...')
    model = torch.load(args.model, map_location=torch.device('cpu'))  # Loading the model itself

    # 2) Load testdata
    print('Loading the test set...')
    df = pd.read_csv(args.data, sep='\t', encoding='utf8', header=0)

    BERT_MODEL = "bert-base-cased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir="./cache")

    print('Finished loading the test set')
    print('===========================')
    print('Classifying:')
    print('===========================')
    num_sents, num_scopes = classify(model, df, tokenizer, args.file, torch.device("cpu"))
    print("{}/{} hasScope: = {} scope".format(num_scopes, num_sents, num_scopes/num_sents))