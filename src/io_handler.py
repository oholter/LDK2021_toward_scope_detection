import pandas as pd
from rdflib import Graph

def read_requirements(path):
    #df = pd.read_csv(path, header=None, sep="|", names=["num", "text"], encoding="utf8")
    df = pd.read_csv(path, header=1, sep="\t", names=['sec', 'req', 'sent_num', 'sent'], encoding="utf8")
    df = df.dropna()
    return df

def read_gazetteer(path):
    df = pd.read_csv(path, header=0, sep="|", encoding="utf8", index_col=False)
    return df

def read_ontology(path):
    df = pd.read_csv(path, header=0, sep=",", encoding="utf8")
    return df

def read_ontology_graph(path):
    g = Graph()
    g.parse(path, format="turtle")
    return g

def write_ontology_graph(graph, path):
    graph.serialize(destination=path, format='turtle')

def write_gazetteer(df, path):
    if not df.empty:
        df.to_csv(path, sep=',', index=False, header=True, encoding='utf8')

def read_objects(path):
    df = pd.read_csv(path, header=0, sep=",", encoding="utf8")
    #df = pd.read_csv(path, header=0, sep=":", encoding="utf8")
    return df


if __name__ == "__main__":
    #df = read_requirements("requirements.txt")
    #df = read_gazetteer("terms.txt")
    #df = read_objects("object.txt")
    df = read_ontology("../req_onto/terms.txt")

    print(df.head())