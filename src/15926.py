from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re


def query():
    """
    Query for all classes that are subClassOf Equipment
    <http://data.15926.org/rdl/RDS8615020> : EQUIPMENT CLASS

    """
    sparql = SPARQLWrapper("http://192.236.179.169/sparql")
    query = """
    SELECT distinct ?concept ?label
    WHERE
    {
        ?concept rdfs:subClassOf+ <http://data.15926.org/rdl/RDS8615020> ;
            rdfs:label ?label .
    }
    """
    print(query)
    #exit()
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    return results_df


def query2():
    """
    Query for all classes that are subClassOf Artefact
    <http://data.1592.org/rdl/RDS201644>: Artefact Class
    <http://data.15926.org/rdl/RDS422594>: Artefact

    """
    sparql = SPARQLWrapper("http://192.236.179.169/sparql")
    #    ?concept rdfs:subClassOf+ <http://data.15926.org/rdl/RDS201644> ;
    query2 = """
    SELECT distinct ?concept ?label
    WHERE
    {
         ?concept rdfs:subClassOf+ <http://data.15926.org/rdl/RDS422594> ;
            rdfs:label ?label .
    }
    """
    print(query2)
    #exit()
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    return results_df


def write_results(file, results_df):
    #print(results_df.columns)
    concepts = []

    for index, row in results_df.iterrows():
        label = row['label.value'].lower()
        if re.search("class$", label):
            label = label[:-6].strip()
        if re.search("^asme", label):
            label = label[12:].strip()

        match = re.search("(.*)for asme.*", label)
        if match:
            #print(label)
            label = match.group(1).strip()
            #print(label)

        #identifier = row['concept.value']
        concepts.append(label)


    #exit()
    #f = open(file, 'w')
    with open(file, 'w') as F:
        F.write("Candidate|identifier\n")
        concepts = set(concepts) # remove many duplicates
        for concept in concepts:
            print(concept)
            F.write(concept)
            F.write("|")
            F.write('\n')





if __name__ == "__main__":
    results_df = query2()
    write_results("../15926.txt", results_df)
