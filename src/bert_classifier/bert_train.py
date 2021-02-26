"""
Training a Bert model with a linear layer on predicting requirement scope presence
@author: Ole Magnus Holter
"""

import sys
sys.path.append("..")
sys.path.append(".")

import logging
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from argparse import ArgumentParser
from tqdm import tqdm
from torchtext.data import Field, Example, Dataset, Iterator
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score




def train(model, optimizer, scheduler, iterator, val_loader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        correct = 0
        tot = 0
        tot_train_loss = 0
        for batch in tqdm(iterator):
            text = batch.text
            Y = torch.LongTensor(batch.label)
            X = torch.LongTensor(text)
            X.to(device)
            Y.to(device)

            # X: <batch_size, seq_len>,
            loss, y_pred = model(X, labels=Y)[:2]

            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            #print(y_pred)
            correct += ((y_pred.argmax(1) == Y).sum()).item()
            tot += len(batch)
            tot_train_loss += loss.item()

        acc = correct / tot
        avg_train_loss = tot_train_loss / len(iterator)
        logging.info('[%d] avg_loss: %.5f, acc: %.5f', epoch + 1, avg_train_loss, acc)
    logging.info("finished training\n")


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


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(levelname)s :: %(message)s", level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("-e", "--epochs", default=4, action='store', type=int)  # paper suggests 2-4
    parser.add_argument("--train", default="/home/ole/src/scope_detection/data/training_data/training_data_majority17221.tsv")
    parser.add_argument("--gold", default="/home/ole/src/scope_detection/data/gold_data/all_samples_gold.tsv")
    parser.add_argument("--save", action="store", default='bert_large_majority2.bin')
    parser.add_argument("--lr", action="store", default=3e-5, type=float)  # paper suggests [2e-5, 5e-5]
    parser.set_defaults(full_finetuning=True)
    parser.add_argument("--full_finetuning", action="store_true", dest="full_finetuning")
    parser.add_argument("--eps", action="store", default=1e-8, type=float)  # 1e-6 default
    args = parser.parse_args()

    print("Using the following parameters:\n ")
    for arg in vars(args):
        print(arg.upper()+": ", getattr(args, arg))
    print("\n")

    seed = 42
    torch.manual_seed(seed)  # improve reproducibility
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    BERT_MODEL = "bert-base-cased"

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2, cache_dir="./cache",
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, cache_dir="./cache")
    model = model.to(device)

    MAX_SEQ_LEN = 32  # Todo: This number should probably be higher (128?)
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    df = pd.read_csv(args.train, sep='\t', header=0, encoding='utf8', names=['sec', 'req', 'sent_num', 'sent', 'label'])
    # removing ['num', 'req_id'] for classification
    df = df[['sent', 'label']]

    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                       fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.long)
    fields = [('text', text_field), ('label', label_field)]

    dataset = []
    for n, entry in df.iterrows():
        # print(entry)
        example = Example.fromlist(entry, fields)
        dataset.append(example)
    dataset = dataset
    dataset = Dataset(dataset, fields)
    train_data, val_data = dataset.split(split_ratio=0.9)

    train_iter = Iterator(train_data, batch_size=32, train=True,
                               shuffle=True)
    val_iter = Iterator(val_data, batch_size=1, shuffle=False)



    if args.full_finetuning:
        param_optimizer = list(model.named_parameters())  # all the model's parameters
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        # the BeretForSequenceClassification is a BertModel with
        # a Linear-layer for classification
        # here we train only the parameters in the linear layer
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]


    # adam optimizer with weight decay fix
    OPTIM = AdamW(
        optimizer_grouped_parameters,
    #    model.parameters(),
        lr=args.lr,
        eps=args.eps
    )

    # linear decay of learning rate as suggested in the paper
    total_steps = len(train_iter) * args.epochs
    SCHEDULER = get_linear_schedule_with_warmup(
        OPTIM,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train(model, OPTIM, SCHEDULER, train_iter, val_iter, args.epochs, device)

    logging.info("Evaluating model on VAL set")
    model.eval()
    acc, report, p_mak, r_mak, f_mak = evaluate(model, val_iter, device)
    print(report)

    if args.gold:
        gold_df = pd.read_csv(args.gold, sep='\t', header=0)
        gold_df = gold_df[['sent', 'label']]  # 300 are corrected
        gold_dataset = []
        for n, entry in gold_df.iterrows():
            example = Example.fromlist(entry, fields)
            gold_dataset.append(example)
        gold_dataset = Dataset(gold_dataset, fields)
        gold_iter = Iterator(gold_dataset, batch_size=1, shuffle=False)
        model.eval()
        logging.info("Evaluating model on GOLD set")
        acc, report, p_mak, r_mak, f_mak = evaluate(model, gold_iter, device)
        print(report)


    if args.save:
        logging.info("Saving the model to %s...", args.save)
        torch.save(model, args.save)