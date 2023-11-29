import logging
import pickle
import time
import uuid

import pandas as pd
import pymongo
import sklearn

import dbconnector
from config import Config
from dbconnector import save_model_to_db, execute_sparql_query


def save_training_data_to_db(training_data: pd.DataFrame, client, db, dbconnection, training_data_uuid):
    pickled_data = pickle.dumps(training_data)
    myclient = pymongo.MongoClient(client)
    mydb = myclient[db]
    connection = mydb[dbconnection]
    info = connection.insert_one({training_data_uuid: pickled_data, 'name': training_data_uuid, 'created_time': time.time()})
    logging.info(str(info.inserted_id) + ' saved successfully!')

    details = {
        'inserted_id': info.inserted_id,
        'data_name': training_data_uuid,
        'created_time': time.time()
    }

    return details


# TODO Auslagern des Trainings und aufteilen in Abspeichern. Nutzer sollten selbst trainieren
def train_model(model: sklearn.base.ClassifierMixin, training_data: pd.DataFrame, label: str):
    X = training_data.drop(label, axis=1)
    y = training_data[label]
    model = model.fit(X, y)
    training_data_uuid = "TrainingData-" + str(uuid.uuid1())
    save_training_data_to_db(
        training_data=training_data,
        client=Config.mongodb_client,
        db=Config.mongodb_database,
        dbconnection="data",
        training_data_uuid=training_data_uuid,
    )
    model_uuid = save_model(model)
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:type 
        <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#TrainingData>;
        festo:trained <http://www.semanticweb.org/kidz/festo#%s> .}"""
    execute_sparql_query(query_template % (training_data_uuid, model_uuid))
    return model_uuid, training_data_uuid


def save_model(model: sklearn.base.ClassifierMixin):
    algorithm_type = type(model).__name__
    model_uuid = "Model-" + algorithm_type + "-" + str(uuid.uuid1())
    logging.info(model_uuid + " created!")

    save_model_to_db(
        model=model,
        client=Config.mongodb_client,
        db=Config.mongodb_database,
        dbconnection="models",
        model_name=model_uuid,
    )

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA {
    <http://www.semanticweb.org/kidz/festo#%s> festo:type 
    <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model>.}"""
    execute_sparql_query(query_template % model_uuid)
    #Todo Magic String entfernen
    dbconnector.connect_model_to_training_run(model_uuid, "TR1")
    return model_uuid
