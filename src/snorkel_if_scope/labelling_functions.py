import sys
sys.path.append("../")
sys.path.append("./")

from spacy.matcher import Matcher
from spacy.util import filter_spans

from src.utils import ABSTAIN, SCOPE, NOT_SCOPE, SpacyDoc, lemmatize_word, contains_chunk, onto_subclass_of
from src.io_handler import read_gazetteer, read_ontology_graph, read_requirements

import pandas as pd
import re
from snorkel.labeling import labeling_function, PandasLFApplier, LFAnalysis, filter_unlabeled_dataframe
from snorkel.analysis import get_label_buckets
from snorkel.labeling.model import MajorityLabelVoter, LabelModel
from snorkel.utils import probs_to_preds

from sklearn.metrics import classification_report


iso15926_gazetteer_path = "data/15926_all.txt"
filtered_termostat_gazetteer_path = "data/wn_list.txt"
wv_gazetteer_path = "data/terms_wv.txt"


#gazetteer = read_gazetteer(gazetteer_path)
iso15926_gazetteer = read_gazetteer(iso15926_gazetteer_path)
#graph = read_ontology_graph(ontology_graph_path)
termostat_gazetteer = read_gazetteer(filtered_termostat_gazetteer_path)
wv_gazetteer = read_gazetteer(wv_gazetteer_path)

@labeling_function()
def has_equipment_iso15926(x):
    req = x.sent
    spdoc = SpacyDoc.get_instance()
    doc = spdoc.get_doc(req)
    stopwords = spdoc.get_stopwords()

    for i, chunk in enumerate(doc.noun_chunks):
        tokens = chunk.text.lower().split(" ")
        tokens_without_stopwords = [lemmatize_word(token) for token in tokens if token not in stopwords]
        while tokens_without_stopwords:
            label = contains_chunk(tokens_without_stopwords, iso15926_gazetteer)
            if label == SCOPE:
                #print(chunk)
                return SCOPE
            tokens_without_stopwords.pop(0)
    return ABSTAIN  # If I return ABSTAIN, performance drops considerably, but I imagine that it will not generalize as much?
    #return NOT_SCOPE


@labeling_function()
def has_equipment_termostat(x):
    req = x.sent
    spdoc = SpacyDoc.get_instance()
    doc = spdoc.get_doc(req)
    stopwords = spdoc.get_stopwords()

    found = False
    for i, chunk in enumerate(doc.noun_chunks):
        tokens = chunk.text.lower().split(" ")
        tokens_without_stopwords = [lemmatize_word(token) for token in tokens if token not in stopwords]
        while tokens_without_stopwords:
            label = contains_chunk(tokens_without_stopwords, termostat_gazetteer)
            if label == SCOPE:
                found = True
                break
                #print(chunk)
                #return SCOPE
            tokens_without_stopwords.pop(0)
    #return ABSTAIN  # If I return ABSTAIN, performance drops considerably, but I imagine that it will not generalize as much?
    #return NOT_SCOPE
    if found:
        return ABSTAIN
    else:
        return NOT_SCOPE


@labeling_function()
def has_equipment_wv(x):
    req = x.sent
    spdoc = SpacyDoc.get_instance()
    doc = spdoc.get_doc(req)
    stopwords = spdoc.get_stopwords()

    for i, chunk in enumerate(doc.noun_chunks):
        tokens = chunk.text.lower().split(" ")
        tokens_without_stopwords = [lemmatize_word(token) for token in tokens if token not in stopwords]
        while tokens_without_stopwords:
            label = contains_chunk(tokens_without_stopwords, wv_gazetteer)
            if label == SCOPE:
                #print(chunk)
                return SCOPE
            tokens_without_stopwords.pop(0)
    return ABSTAIN  # If I return ABSTAIN, performance drops considerably, but I imagine that it will not generalize as much?
    #return NOT_SCOPE


@labeling_function()
def too_large(x):
    req = x.sent
    if len(req) > 1000:
        return NOT_SCOPE
    else:
        return ABSTAIN


@labeling_function()
def contains_colon(x):
    req = x.sent
    if ':' in req:
        return NOT_SCOPE
    else:
        return ABSTAIN


@labeling_function()
def verb_shall_chunks(x):
    """typically an action or procedure"""
    req = x.sent
    nlp = SpacyDoc.get_instance().get_nlp()

    p1 = [{'POS' : 'VERB'},
          {'LOWER' : 'shall'}]

    matcher = Matcher(nlp.vocab)
    matcher.add("for phrase", None, p1)

    doc = SpacyDoc.get_instance().get_doc(req)
    # call the matcher to find matches
    matches = matcher(doc)
    verb_chunks = [doc[start:end] for _, start, end in matches]

    filtered_vc = filter_spans(verb_chunks)
    #print(filtered_vc)
    return NOT_SCOPE if filtered_vc else ABSTAIN




@labeling_function()
def contains_For(x):
    """typical pattern: For [equipment] """
    req = x.sent
    return SCOPE if re.search(r'For', req) else ABSTAIN


@labeling_function()
def contains_This(x):
    """Central subject is typically unknown in a phrase starting with This"""
    req = x.sent
    return NOT_SCOPE if re.search(r'This', req) else ABSTAIN

#@labeling_function()
#def contains_report(x):
#    """many requirements are about reports"""
#    req = x.sent
#    return NOT_SCOPE if re.search(r'\s(report|survey)s?\s', req, flags=re.I) else ABSTAIN


@labeling_function()
def contains_scope_words(x):
    req = x.sent
    tokens = ['shall be capable of',
              'shall be designed',
              'shall be tested',
              ]

    for tok in tokens:
        pattern = tok + "\W"
        if re.search(pattern, req, flags=re.I):
        #if tok + " " in req:
            return SCOPE

    return ABSTAIN

@labeling_function()
def contains_non_skope_words(x):
    """tried to combine a short list of common terms in non-req sents"""
    req = x.sent
    tokens = ['report',
              'survey',
              'shall include',  # documentation and methods
              'describe',  # document and concepts
              'description',
              'drawing',
              'parameters',
              'parameter',
              'results',  # survey or testing
              'examination',
              'include',
              'include:',
              'shall be taken as',  # definition
              'shall be taken as:',
              'carried out',
              'shall be used to',
              'shall cover',
              'be based on',
              'be performed',
              'evaluate',
              'calculation',
              'calculations',
              'analysis shall',
              'criteria',  # a value
              'based upon',  # a value/calculation
              'determined',  # a value
              'details',  # something to note down or taken care of
              'references',  # not reference (reference standard)
              'inspection',
              'testing shall',
              'hence',  # explanation
              'procedure',
              'procedures',
              'review',
              'testing',
              ]
    for tok in tokens:
        pattern = tok + "\W"
        if re.search(pattern, req, flags=re.I):
        #if tok + " " in req:
            return NOT_SCOPE

    return ABSTAIN


def write_training_data(data, labels, file_path):
    data['label'] = labels
    data.columns = ['sec', 'req', 'sent_num', 'sent', 'label']
    with open(file_path, 'w') as F:
        F.write("sec\treq\tsent_num\tsent\tlabel\n")
        for idx, row in data.iterrows():
            # print(row)
            F.write(str(idx))
            F.write("\t")
            F.write(str(row['sec']))
            F.write("\t")
            F.write(str(row['req']))
            F.write("\t")
            F.write(str(row['sent_num']))
            F.write("\t")
            F.write(str(row["sent"]))
            F.write("\t")
            F.write(str(row['label']))
            F.write("\n")


def evaluate(gold_filtered, pred_labels):
    y = gold_filtered['label']
    y_pred = pred_labels
    report = classification_report(y, y_pred)
    print(report)
    print("Remember that this evaluation disregards all the ones that were labeled ABSTAIN")




if __name__ == "__main__":
    # Avoid printing truncated DataFrames
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    #requirements_path = "/home/ole/src/scope_detection/requirements.txt"
    gold_path = "tsv/all_samples_gold.tsv"
    requirements_path = "tsv/all_samples_removed.tsv"


    df = read_requirements(requirements_path)
    df_gold = pd.read_csv(gold_path, sep='\t', header=0)
    df_gold.columns = ['sec', 'req', 'sent_num', 'sent', 'label']
    df_test = df_gold[['sec', 'req', 'sent_num', 'sent', 'label']][:200]
    df_train = df
    Y_test = df_test.label.values


    lfs = [ has_equipment_termostat,
            has_equipment_iso15926,
            has_equipment_wv,
            contains_colon,
            contains_For,
            contains_scope_words,
            contains_non_skope_words
            ]

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df_train)
    L_test = applier.apply(df=df_test)

    print(LFAnalysis(L=L_train, lfs=lfs).lf_summary())

    buckets = get_label_buckets(L_train[:, 1], L_train[:, 2])
    #print(df_train.iloc[buckets[(SCOPE, ABSTAIN)]].sample(10, random_state=1))

    majority_model = MajorityLabelVoter()
    preds_train = majority_model.predict(L=L_train)
    majority_probs_train = majority_model.predict_proba(L=L_train)

    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)
    label_probs_train = label_model.predict_proba(L=L_train)

    majority_acc = majority_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
    print(f"{'Majority Vote Accuracy:':<25} {majority_acc * 100:.1f}%")

    label_model_acc = label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")['accuracy']
    print(f"{'Label model Accuracy:':<25} {label_model_acc * 100:.1f}%")

    #df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=label_probs_train, L=L_train)
    df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(X=df_train, y=majority_probs_train, L=L_train)
    pred_train_filtered = probs_to_preds(probs=probs_train_filtered)

    # output the gold to be able to evaluate
    #probs_gold = label_model.predict_proba(L=L_test)
    probs_gold = majority_model.predict_proba(L=L_test)
    df_gold_filtered, probs_gold_filtered = filter_unlabeled_dataframe(X=df_gold, y=probs_gold, L=L_test)
    pred_gold_filtered = probs_to_preds(probs=probs_gold_filtered)

    if not df_train_filtered.empty:
        write_training_data(df_train_filtered, pred_train_filtered, "training_data_majority240221.tsv")

    # At last, a direct evaluation of the labelling function
    evaluate(df_gold_filtered, pred_gold_filtered)
