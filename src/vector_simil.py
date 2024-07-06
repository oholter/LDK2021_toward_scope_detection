import pandas as pd
import gensim.downloader as api
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from functools import reduce
import matplotlib.pyplot as plt
from src.utils import SpacyDoc
from spacy.matcher import Matcher
from spacy.util import filter_spans

equipments = ["riser", "accumulator", "engine", "compressor", "blower", "controller", "generator", "turbine",
              "pipeline", "sensor", "pump", "vessel", "valve", "bolt", "cable", "clamp", "connector", "cooler",
              "fan", "filter", "fitting", "flange", "gearbox", "joint", "pipe", "nut", "pump", "reflector", "tube", "ship"
              ]

def cosine(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))


class VectorComparator:
    __instance = None
    __model = None
    __equipment_wv = None
    __dim = 0

    @staticmethod
    def get_instance():
        if VectorComparator.__instance is None:
            VectorComparator()
        return VectorComparator.__instance

    def __init__(self, dim=100):
        if self.__instance != None:
            raise Exception("This is a singleton")

        VectorComparator.__dim = dim
        VectorComparator.__instance = self
        VectorComparator.__model = api.load("glove-wiki-gigaword-100")
        VectorComparator.__equipment_wv = self.calculate_equipment_wv()

    def get_model(self):
        return self.__model

    def calculate_equipment_wv(self):
        equipment_wv = np.zeros(self.__dim)
        for equipment in equipments:
            equipment_wv += self.__model[equipment]
        equipment_wv /= len(equipments)
        return equipment_wv

    def get_equipment_similarity(self, string):
        terms = string.strip().lower().split(" ")
        vec = np.zeros(self.__dim)
        num_terms = 0

        # using the average vector
        for term in terms:
            num_terms += 1
            if term in self.__model:
                vec += self.__model[term]

        if vec.any():  # not the zero vector
            return cosine(vec, self.__equipment_wv)
        else:
            return -1


def write_most_similar(n):
    print(model.most_similar(positive=[equipment_wv], topn=n))
    most_simil = model.most_similar(positive=[equipment_wv], topn=n)
    with open('data/terms_wv.txt', 'w') as F:
        F.write("Candidate\n")
        for word, score in most_simil:
            F.write(word)
            F.write('\n')


def create_clusters(data, num_clusters=7):
    #dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))
    hc = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
    agglo = hc.fit_predict(data)
    return agglo


def print_clusters(agglo, data, model):
    clusters = [[] for i in range(max(agglo) + 1)]
    n = 0
    for clustno, word in zip(agglo, data):
        clusters[clustno].append(word)
        n += 1

    print("Tot: {} elem".format(n))

    n = 0
    for cluster in clusters:
        print("\nCluster {}".format(n))
        n += 1
        term_vectors = [get_term_vector(w, model) for w in cluster]
        sum_vector = np.zeros(100)

        for vec, last in term_vectors:
            sum_vector += last

        central_concept = model.most_similar(positive=[sum_vector], topn=1)[0]
        print("Title: {}".format(central_concept[0]))
        for i in cluster:
            print(i)

    print("\nTot elements: {}".format(n))


def find_titles(agglo, words, model):
    clusters = [[] for i in range(max(agglo) + 1)]
    central_concepts = []
    n = 0
    for clustno, word in zip(agglo, words):
        clusters[clustno].append(word)
        n += 1

    n = 0
    for cluster in clusters:
        sum_vector = add_vectors([model[w] for w in cluster])
        central_concept = model.most_similar(positive=[sum_vector], topn=1)[0]
        central_concepts.append(central_concept)

    return np.array(central_concepts)[:, 0]

    # todo: must fix plot


def plot_cluster_data(vectors, agglo, central_concepts):
    pca = PCA(n_components=2)
    result = pca.fit_transform(vectors)
    plt.clf()

    n = 0
    clusters = [[] for i in range(max(agglo) + 1)]
    for clustno, vector in zip(agglo, result):
        clusters[clustno].append(result[n])
        n += 1

    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1])
        concept = central_concepts[i]
        # concept = concept.split("#")[1]
        plt.annotate(concept, xy=(cluster[0, 0], cluster[0, 1]))
    plt.show()


def add_vectors(vectors):
    sum_vec = reduce(lambda x, y: x + y, vectors)
    # ave_vec = sum_vec / len(vectors)
    # return ave_vec
    return sum_vec


def get_term_vector(terms, model):
    vec = np.zeros(dim)
    last_vec = np.zeros(100)
    if terms[-1] in model:
        last_vec = model[terms[-1]]
    num_terms = 0
    for term in terms:
        num_terms += 1
        if term in model:
            vec += model[term]

    if num_terms > 0:
        words.append(terms)
        return vec, last_vec
    else:
        return None, None


def contains_verb(term):
    """If the term contains a verb it is probably NOT an equipment"""
    nlp = SpacyDoc.get_instance().get_nlp()

    p1 = [{'POS': 'VERB'}]

    matcher = Matcher(nlp.vocab)
    matcher.add("contains_verb", None, p1)

    doc = SpacyDoc.get_instance().get_doc(term)
    # call the matcher to find matches
    matches = matcher(doc)
    verb_chunks = [doc[start:end] for _, start, end in matches]

    filtered_vc = filter_spans(verb_chunks)
    if filtered_vc:
        pass
        #print("{} contains a verb".format(term))
        #print(filtered_vc)
    return True if filtered_vc else False



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s :: %(levelname)s :: %(message)s",
                        level=logging.INFO)
    dim = 100
    model = api.load("glove-wiki-gigaword-100")

    comparator = VectorComparator.get_instance()
    print(comparator.get_equipment_similarity("pipeline"))


    equipment_wv = np.zeros(dim)
    for equipment in equipments:
        equipment_wv += model[equipment]
    equipment_wv /= len(equipments)

    #
    # A generative approach - This seems to actually improve performance of the method
    write_most_similar(n=100)
