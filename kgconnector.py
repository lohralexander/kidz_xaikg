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
    uuid.uuid1()
