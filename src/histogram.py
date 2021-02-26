"""
This script counts number of tokens of incorrectly labelled sentences and
creates a histogram
"""

from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from nltk import word_tokenize








if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(levelname)s :: %(message)s", level=logging.INFO)
    parser = ArgumentParser()
    #parser.add_argument("--gold", default="/home/ole/src/scope_detection/data/gold_data/all_samples_gold.tsv")
    parser.add_argument("--gold", default="/home/ole/src/scope_detection/data/gold_data/equinor_samples.tsv")
    parser.add_argument("--data", default="/home/ole/src/scope_detection/data/other/equinor_samples_classified.tsv")
    args = parser.parse_args()

    # Avoid printing truncated DataFrames
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    # read data
    df_gold = pd.read_csv(args.gold, encoding='utf8', header=0, sep='\t', dtype='str')
    df_data = pd.read_csv(args.data, encoding='utf8', header=0, sep='\t', dtype='str')

    duplicates = pd.merge(df_gold, df_data, how='inner', left_on=['sec', 'req', 'sent_num', 'sent', 'label'],
                          right_on=['sec', 'req', 'sent_num', 'sent', 'label'], left_index=True)

    different = df_data.drop(duplicates.index)
    #print(different)
    logging.info("Length of different: {}".format(len(different)))


    xs = []
    for i, row in different.iterrows():
        sent = row['sent']
        toks = word_tokenize(sent)
        xs.append(len(toks))
    #print(xs)

    ys = []
    for i, row in df_gold.iterrows():
        sent = row['sent']
        toks = word_tokenize(sent)
        ys.append(len(toks))

    bins = np.linspace(5, 32, 10)
    plt.hist(xs, bins, density=True, alpha=0.5, label='incorrectly labelled')
    plt.hist(ys, bins, density=True, alpha=0.5,  label='all sentences')
    plt.xlabel('Sentence length')
    plt.ylabel('Fraction')
    plt.show()
    #plt.savefig("histogram.png")




