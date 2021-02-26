from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
import re



        #"PREFIX  rdfs:     <http://www.w3.org/2000/01/rdf-schema#>" \
        #+ "PREFIX  rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" \
            #+ "FILTER(CONTAINS(LCASE(?itemLabel)," + equipment + "@en)) ." \
            #"OPTIONAL { ?entity rdfs:label ?label ." \
                        #"FILTER((LANG(?label)) = 'en')}" \

def query(equipment):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = \
        "SELECT ?entity ?entityLabel" \
        "WHERE" \
        "{" \
            "?entity wdt:P279 wd:Q16798631 ." \
            "SERVICE wikibase:label { bd:serviceParam wikibase:language 'en' }" \
        "} LIMIT 100"
    sparql.setQuery(query)

    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    print(results_df)


def query2():
    """
    Query for all classes that are subClassOf Equipment
    and NOT subClassOf Physical Quanitity
    wdt:P279 = subClassOf
    wd:Q107715 = Physical Quantity
    wd:Q16798631 = Equipment

    comments: The query takes some time. 1 minute?
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    #sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query2 = """
    SELECT ?item ?itemLabel
    WHERE
    {
        ?item wdt:P279+ wd:Q16798631 .
        FILTER NOT EXISTS {
            ?item wdt:P279+ wd:Q107715 .
        }
        SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" }
    }
    """
    #print(query2)
    #exit()
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    return results_df

def query3():
    """
    Query for all classes that are subClassOf Tool
    wdt:P279 = subClassOf
    wd:Q107715 = Physical Quantity
    wd:Q16798631 = Equipment
    wd:Q39546 = Tool

    comments: The query takes some time. 1 minute?
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    #sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query2 = """
    SELECT ?item ?itemLabel
    WHERE
    {
        ?item wdt:P279+ wd:Q39546 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" }
    }
    """
    #print(query2)
    #exit()
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    return results_df

# todo: Not implemented
def dbpedia_query():
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query2 = """
    SELECT ?item ?itemLabel
    WHERE
    {
        ?item wdt:P279+ wd:Q16798631 .
        SERVICE wikibase:label { bd:serviceParam wikibase:language \"en\" }
    }
    """
    sparql.setQuery(query2)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    results_df = pd.io.json.json_normalize(results['results']['bindings'])
    #print(results_df)
    return results_df




def write_results(file, results_df):
    #print(results_df.columns)
    wd_identifier = r"[Qq]\d\d\d\d\d\d\d"
    f = open(file, 'w')
    f.write("Candidate|identifier\n")
    for index, row in results_df.iterrows():
        #print(row)
        value = row['itemLabel.value']
        identifier = row['item.value']
        if not re.findall(wd_identifier, value):
            print(value.lower())
            f.write(value.lower())
            f.write("|")
            f.write(identifier)
            f.write('\n')
    f.close()





if __name__ == "__main__":
    results_df = query3()
    write_results("../equipment2.txt", results_df)
