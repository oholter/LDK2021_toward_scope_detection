import pandas as pd
from src.io_handler import read_requirements
from tqdm import tqdm


if __name__ == "__main__":
    # Avoid printing truncated DataFrames
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    df_all = pd.read_csv("../data/raw_text/all_samples_removed.tsv", sep='\t', header=0, encoding='utf8', dtype='str')
    df_samples = pd.read_csv("../data/gold_data/all_samples_dnv.tsv", sep='\t', header=0, encoding='utf8', dtype='str')

    #sent_filter = ~df_all[['sent']].isin(df_samples[['sent']])
    #req_filter = ~df_all[['req']].isin(df_samples[['req']])
    #req_filter.columns = ['sent']
    #filter = sent_filter & req_filter
    #print(filter)
    #print(df_all[filter])



    duplicates = pd.merge(df_samples, df_all, how='inner', left_on=['sec', 'req', 'sent_num', 'sent'],
                          right_on=['sec', 'req', 'sent_num', 'sent'], left_index=True)

    print("num_duplicates: {}".format(len(duplicates['sent'])))

    new_df_all = df_all.drop(duplicates.index)
    #print(df_all)
    print("len new all: {}".format(len(new_df_all)))
    #print(duplicates)
    new_df_all.to_csv("../data/raw_text/all_samples_removed.tsv", index=False, encoding='utf8', sep='\t', header=True)


