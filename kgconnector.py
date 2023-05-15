import uuid

from SPARQLWrapper import SPARQLWrapper, JSON

from config import Config


def get_training_data():
    sparql = SPARQLWrapper(Config.repository)
    sparql.setQuery("""
                PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
            SELECT * { 
                ?experiment a festo:Experiment ;
                            festo:pressure ?pressure;
                            festo:testedCylinder ?cylinder;
                            festo:result ?result.
                ?cylinder festo:weight ?weight.
                ?cylinder festo:hasMaterial ?material.
                ?cylinder festo:hasBottom ?bottom
            }
            """)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def save_prediction():
    unique_identifier = "Pred" + str(uuid.uuid1())
    print(unique_identifier)
    sparql = SPARQLWrapper(Config.repository_update)
    query_string = (
                """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {<http://www.semanticweb.org/kidz/festo#""" + unique_identifier + """>
    dc:result "Prediction 1" ; dc:Station "Ward 2" .}""")
    sparql.setQuery(query_string)
    sparql.method = 'POST'
    sparql.query()


if __name__ == '__main__':
    save_prediction()
