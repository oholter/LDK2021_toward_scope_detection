import logging
from nltk.corpus import wordnet as wn
import pandas as pd
from src.io_handler import read_gazetteer

logging.basicConfig(format="%(asctime)s :: %(levelname)s :: %(message)s",
                    level=logging.INFO)

logger = logging.Logger("__name__")


class object_chekcer:
    object_hypos = None

    def is_object(self, string):

        if self.object_hypos is None:
            logger.info("building list of synsets")
            #object = wn.synsets('object')
            #object = wn.synsets('equipment')
            #object = wn.synsets('instrumentation')
            #object = wn.synsets('whole')
            object = wn.synsets('artifact')
            object_ss = object[0]
            #print(eq.definition())
            hypos = lambda s:s.hyponyms()
            self.object_hypos = list(object_ss.closure(hypos))

        ss = wn.synsets(string)
        for p in ss:
            if p in self.object_hypos:
                #print(p.definition())
                return True

        return False




if __name__ == "__main__":
    checker = object_chekcer()

    #gazetteer = read_gazetteer("/home/ole/src/scope_detection/terms.txt")
    gazetteer = pd.read_csv("/home/ole/src/scope_detection/terms_from_termostat.txt", encoding='latin1', sep=',')
    #print(gazetteer['Candidate'])
    num_terms = 0
    for i, term in enumerate(gazetteer['Candidate']):
        isobject = checker.is_object(term)
        if isobject:
            num_terms += 1
            print(term)

    print("number of items: {}".format(num_terms))
    #exit(0)



