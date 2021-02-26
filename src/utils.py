from dataclasses import dataclass
import spacy
import numpy as np
from textblob import Word
from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL

ABSTAIN = -1
NEITHER = 0
NOT_SCOPE = NEITHER
SCOPE = 1
CONDITION = 2
DEMAND = 3
UNKNOWN = 4

NSS = {"owl": OWL,
       "rdfs": RDFS,
       "rdf": RDF,
       "req": "http://www.semanticweb.org/ole/ontologies/2020/9/requirements#"}


@dataclass()
class Span:
    start: int = -1
    end: int = -1
    label: int = -1
    confidence: float = 0.0
    text: str = ""
    cls: str = ""


class SpacyDoc():
    nlp = None
    stopwords = None
    doc = None
    __instance = None

    @staticmethod
    def get_instance():
        if SpacyDoc.__instance is None:
            SpacyDoc()
        return SpacyDoc.__instance

    def __init__(self):
        if self.__instance != None:
            raise Exception("This is a singleton")

        SpacyDoc.__instance = self
        SpacyDoc.nlp = spacy.load("en_core_web_sm")
        SpacyDoc.stopwords = SpacyDoc.nlp.Defaults.stop_words

    def get_doc(self, text):
        return self.nlp(text)

    def get_nlp(self):
        return self.nlp

    def get_stopwords(self):
        return self.stopwords


def normalize_tokens(tokens):
    """

    :param tokens: [a list of tokens]
    :return: a string with without stopwords
    """
    spdoc = SpacyDoc.get_instance()
    stopwords = spdoc.get_stopwords()
    tokens_without_stopwords = [lemmatize_word(token) for token in tokens if token not in stopwords]
    words = " ".join(tokens_without_stopwords)
    return words


def lemmatize_word(word):
    w = Word(word)
    lemmatized_word = w.lemmatize()
    return lemmatized_word


def contains_chunk(chunk, gazetteer):
    """ chunk: list of words ["word, "another"]
    returns:
        -1 (not found) or label
        >>> contains_chunk(["pipeline"])
        1
        >>> contains_chunk(["pipeline", "system"])
        1
        >>> contains_chunk(["stail"])
        -1
        >>> contains_chunk(["pipeline", "problem"])
        -1
        >>> contains_chunk(["testing"])
        1
    """

    if type(chunk) == list:
        term = " ".join(chunk)
    else:
        #print(chunk)
        w = Word(chunk)
        w = w.lemmatize()
        #term = chunk
        #print(w)
        term = w
    #print(gazetteer['Candidate'])
    #print(term)
    found = gazetteer[gazetteer['Candidate'] == term]
    #if not found.empty:
    #    print("found" + " : " + term)

    if found.empty:
        return -1
    else:
        return SCOPE  # assuming the gazetteer is an equipment list (!)

def get_subclasses(type, graph):
    w = Word(type)
    w = w.lemmatize()
    term = w.capitalize()

    q = "SELECT DISTINCT ?subClass ?label \n " \
        "WHERE { \n" \
        "?class rdf:type owl:Class . \n " \
        "?class rdfs:label \"" + term + "\" .\n" \
        "?subClass rdfs:subClassOf+ ?class . \n" \
        "?subClass rdfs:label ?label .\n" \
        "}"
    #print(q)
    subclasses = []
    for row in graph.query(q, initNs=NSS):
        subclasses.append(row.label.lower())
    #print("get_subclasses: {}".format(subclasses))
    return subclasses


def remove_duplicate_spans(list_of_spans):
    """Just create a new list with no duplicate spans
    two spans are equal if the start/end indexes are equal and the labels are equal
    the labelling and the class type differ so this is not regarded """
    new_spans = []
    for span in list_of_spans:
        found = False
        for new_span in new_spans:
            if span.label == new_span.label and span.start == new_span.start and span.end == new_span.end:
                found = True
                break

        if not found:
            new_spans.append(span)

        #else:
            #print('removed: {}'.format(span))

    return new_spans



def num_to_class_text(labels):
    label_list = []
    for i, label in enumerate(labels):
        if label == SCOPE:
            if i > 0 and "S" in label_list[i-1]:
                label_list.append("I-S")
            else:
                label_list.append("B-S")
        elif label == CONDITION:
            #print(label_list)
            if i > 0 and "C" in label_list[i-1]:
                label_list.append("I-C")
            else:
                label_list.append("B-C")
        elif label == DEMAND:
            if i > 0 and "D" in label_list[i-1]:
                label_list.append("I-D")
            else:
                label_list.append("B-D")
        else:
            label_list.append("O")

    return label_list



def majority_vote(text, list_of_spans, scope=True, condition=True, demand=True):
    """
    params:
        text: the requirement in textual form
        list_of_spans should be a list of lists
            Each sublist contains the proposed labels/spans from one lf
        scope, condition, demand: True = include these

    Should return all scopes/conditions/demands that are mutually exclusive.
    In case of conflicts, it should consider take the one most frequent
    In case of partly overlaps, it should take the longest span """

    # It should be possible to work with a subset of the labels


    new_list_of_spans = []

    #print(list_of_spans)
    for span_list in list_of_spans:
        new_sublist = []
        for span in span_list:
            if not scope and span.label == SCOPE:
                continue
            elif not condition and span.label == CONDITION:
                continue
            elif not demand and span.label == DEMAND:
                continue
            elif span.label == ABSTAIN:
                continue
            else:
                new_sublist.append(span)
        if new_sublist:
            new_list_of_spans.append(new_sublist)
    #print(new_list_of_spans)
    list_of_spans = new_list_of_spans
    #exit()


    num_labels = 4
    #if not scope:
    #    num_labels -= 1
    #    for span_list in list_of_spans:
    #        for span in span_list:
    #            if span.label == SCOPE:
    #                span.label = ABSTAIN
    #if not condition:
    #    num_labels -= 1
    #    for span_list in list_of_spans:
    #        for span in span_list:
    #            if span.label == CONDITION:
    #                span.label = ABSTAIN
    #if not demand:
    #    num_labels -= 1
    #    for span_list in list_of_spans:
    #        for span in span_list:
    #            if span.label == DEMAND:
    #                span.label = ABSTAIN


    num_lf = len(list_of_spans)
    #print("num_lf: {}".format(num_lf))
    #print("list of spans: {}".format(list_of_spans))


    #this way is used much and has the effect of splitting out commas etc.
    #tokens = text.split(" ")
    doc = SpacyDoc.get_instance().get_doc(text)
    text = " ".join([tok.text for tok in doc])
    tokens = text.split(" ")

    labels = np.zeros([num_lf, len(tokens)]).astype(int)
    votes = np.zeros([num_labels, len(tokens)]).astype(int)

    #print(tokens)

    try:
        for fl_num, span_list in enumerate(list_of_spans):
            for span in span_list:
                for i in range(span.start, span.end):
                    if span.label in (1, 2, 3):
                        labels[fl_num][i] = span.label
    except Exception:
        print("something occured, probably wrong span numbering")
        print(span)

    for fl_num, labels in enumerate(labels):
        for i, label in enumerate(labels):
            if label in (1, 2, 3):  #  ignore abstain, unknown etc
                votes[label][i] += 1


    voted_labels = votes.argmax(axis=0)

    confidence = np.zeros(len(tokens))

    for i, col in enumerate(np.transpose(votes)):
        cnt = votes[voted_labels[i]][i]
        #print(col, voted_labels[i], cnt)
        confidence[i] = cnt / num_lf
        #print(np.count_nonzero(col == voted_labels[i]))


    #print("confidence: {}".format(confidence))

    # remove low confidence labels:
    final_labels = voted_labels
    #for i, label in enumerate(voted_labels):
    #    if confidence[i] <= (1 / num_lf):  # remove if only one function finds this
    #        final_labels[i] = 0

    #print(votes)
    #print(num_to_class_text(voted_labels))
    return num_to_class_text(final_labels), tokens







def get_lowest_level_subclass(classes, graph):
    """
    Checks how many times each of the classes is found among the subclasses of the others.
    The one that is found the most is assumed to be the class of lowest level
    :param classes:
    :param graph:
    :return: lowest level class (string)
    """
    matches = [0]*len(classes)

    for i, type in enumerate(classes):
        subclasses = get_subclasses(type, graph) # get all the subclasses of one of them

        for j, type in enumerate(classes): #check if it is present in the list of subclasses
            if type in subclasses:
                matches[j] += 1

    lowest_level = np.array(matches).argmax()
    return classes[lowest_level]



def subunit_of_type(component_chunk, graph):
    if type(component_chunk) == list:
        term = " ".join(component_chunk)
    else:
        w = Word(component_chunk)
        w = w.lemmatize()
        term = w.capitalize()

    q = "SELECT DISTINCT ?component ?label \n " \
        "WHERE { \n" \
        "?component rdfs:label \"" + term + "\" .\n" \
        "?component rdfs:subClassOf* ?superClass \n . " \
        "?superClass rdfs:subClassOf ?restriction . \n" \
        "?restriction rdf:type owl:Restriction ; \n" \
        "    owl:onProperty req:assembledPartOf ;\n" \
        "    owl:someValuesFrom ?equipment  .\n" \
        "?equipment rdfs:label ?label .\n" \
        "}"
    #print(q)
    classes = []
    for row in graph.query(q, initNs=NSS):
        classes.append(row.label.lower())

    lowest_class = get_lowest_level_subclass(classes, graph)

    #print(classes)

    return lowest_class



def onto_subclass_of(chunk, graph):
    """This uses rdf-lib and ttl-file"""

    if type(chunk) == list:
        term = " ".join(chunk)
    else:
        w = Word(chunk)
        w = w.lemmatize()

        # it is important (!) NOT to capitalize already capitalized terms
        # it can be a capitalized abbreviation such as ISO, DNV-PR-..
        if w[0].isupper():
            term = w
        else:
            term = w.capitalize()

        if "\"" in term:  # the string cannot contain quote (") characters
            term = term.replace("\"", "\\\"")

    q = "select distinct ?superClassLabel where {"\
        "  ?superClass rdfs:label ?superClassLabel ."\
        "  ?subClass rdfs:subClassOf* ?superClass ." \
        "  ?subClass rdfs:label \"" + term + "\" . }"
    #print(q)
    super_classes = []
    try:
        for row in graph.query(q, initNs=NSS):
            super_classes.append(row.superClassLabel.lower())
    except Exception:
        print("Something is wrong with the query:")
        print(q)

    #print(super_classes)
    if 'equipment' in super_classes:
        #print("Equipment is subclass of equipment!!!")
        return "equipment"
    elif 'scale' in super_classes:
        return 'scale'
    elif 'information object' in super_classes:
        return "information object"
    elif 'activity' in super_classes:
        return 'activity'
    elif 'measure' in super_classes:
        return 'measure'
    elif 'physical quantity' in super_classes:
        return 'physical quantity'
    elif 'aspect' in super_classes:
        return 'aspect'
    elif 'subunit' in super_classes:
        return 'subunit'
    elif 'physical object' in super_classes:
        return 'physical object'
    elif 'organisation' in super_classes:
        return 'organisation'
    elif 'location' in super_classes:
        return 'location'
    elif 'system' in super_classes:
        return 'system'
    else:
        return None


def onto_contains_chunk(chunk, ontology):
    """ chunk: list of words ["word, "another"]
    returns:
        None (not found) or type (str)
    """

    if type(chunk) == list:
        term = " ".join(chunk)
    else:
        w = Word(chunk)
        w = w.lemmatize()
        term = w
    found = ontology[ontology['Candidate'] == term]

    if found.empty:
        return None
    else:
        return found['Class'].values[0]


def divide_confidence(spans):
    num_chunks = len(spans)
    for span in spans:
        span.confidence = 1 / num_chunks