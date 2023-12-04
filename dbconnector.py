import logging
import pickle
import time
import uuid

import pandas as pd
import pymongo
from SPARQLWrapper import SPARQLWrapper, JSON

from config import Config

mongo_client = pymongo.MongoClient(Config.mongodb_client)
mongo_database = mongo_client[Config.mongodb_database]


def store_model_kg(model_uuid: str, training_data_uuid: str):
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:type 
        <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#TrainingData>;
        festo:trained <http://www.semanticweb.org/kidz/festo#%s> .}"""
    return execute_sparql_query(query_template % (training_data_uuid, model_uuid))


def save_training_data_to_db(training_data: pd.DataFrame):
    training_data_uuid = "TrainingData-" + str(uuid.uuid1())
    pickled_data = pickle.dumps(training_data)
    connection = mongo_database['data']
    info = connection.insert_one(
        {training_data_uuid: pickled_data, 'name': training_data_uuid, 'created_time': time.time()})
    logging.info(str(info.inserted_id) + ' saved successfully!')

    details = {
        'inserted_id': info.inserted_id,
        'data_name': training_data_uuid,
        'created_time': time.time()
    }

    return training_data_uuid


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


def save_model_to_db(model, client, db, dbconnection, model_name):
    # pickling the model
    pickled_model = pickle.dumps(model)
    connection = mongo_database[dbconnection]
    info = connection.insert_one({model_name: pickled_model, 'name': model_name, 'created_time': time.time()})
    print(info.inserted_id, ' saved with this id successfully!')

    details = {
        'inserted_id': info.inserted_id,
        'model_name': model_name,
        'created_time': time.time()
    }

    return details


def load_saved_object_from_db(object_name, client, db, dbconnection):
    json_data = {}

    # saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)

    # creating database in mongodb
    mydb = myclient[db]

    # creating collection
    mycon = mydb[dbconnection]
    data = mycon.find({'name': object_name})

    for i in data:
        json_data = i
    # fetching model from db
    pickled_model = json_data[object_name]

    return pickle.loads(pickled_model)


def load_saved_objects_from_db(object_name, client, db, dbconnection):
    # saving model to mongoDB
    # creating connection
    myclient = pymongo.MongoClient(client)

    # creating database in mongodb
    mydb = myclient[db]
    collection = mydb[dbconnection]

    query = {"name": {"$regex": object_name}}
    docs = collection.count_documents(query)

    data = collection.find({'name': {"$regex": object_name}})

    print(str(docs) + " Items found!")
    return data


def get_used_model_for_prediction(prediction_uuid: str):
    query_template = """ PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
     PREFIX owl: <http://www.w3.org/2002/07/owl#>
     PREFIX kidzarchitecture: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>
     PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        SELECT ?usedmodel WHERE {<http://www.semanticweb.org/kidz/%s> kidzarchitecture:hasInput ?usedmodel .
            ?usedmodel rdf:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model>.}"""

    sparql = SPARQLWrapper(Config.repository)
    sparql.setQuery(query_template
                    % (prediction_uuid))
    # Setze das gewünschte Rückgabefomat (json, xml, etc.)
    sparql.setReturnFormat(JSON)

    # Führe die SPARQL-Abfrage aus und erhalte das Ergebnis
    results = sparql.query().convert()

    # Verarbeite das Ergebnis (z.B., Ausgabe der Ergebnisse)
    for result in results["results"]["bindings"]:
        print(result["usedmodel"]["value"])
        return result["usedmodel"]["value"]


def execute_sparql_query(query_string: str):
    sparql = SPARQLWrapper(Config.repository_update)
    sparql.setQuery(query_string)
    sparql.method = "POST"
    return sparql.query()


def connect_model_to_training_run(model_uii, training_run):
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA 
    {<http://www.semanticweb.org/kidz/festo#%s> festo:hasOutput <http://www.semanticweb.org/kidz/festo#%s>.}"""
    execute_sparql_query(query_template % (training_run, model_uii))
