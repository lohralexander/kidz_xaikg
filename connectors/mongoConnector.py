import logging
import pickle
import time
import uuid

import pandas as pd

from connectors.dbconnector import get_mongo_collection, execute_sparql_query_write, store_model_kg, \
    connect_model_to_training_run


def save_training_data_to_db(training_data: pd.DataFrame):
    training_data_uuid = "TrainingData-" + str(uuid.uuid1())
    pickled_data = pickle.dumps(training_data)
    connection = get_mongo_collection('data')
    info = connection.insert_one(
        {training_data_uuid: pickled_data, 'name': training_data_uuid, 'created_time': time.time()})
    logging.info(str(info.inserted_id) + ' saved successfully!')

    # for feature in [*training_data]:

    return training_data_uuid


def save_model_to_db(model, training_data_uuid: str):
    pickled_model = pickle.dumps(model)
    algorithm_type = type(model).__name__
    model_uuid = "Model-" + algorithm_type + "-" + str(uuid.uuid1())
    collection = get_mongo_collection('models')
    info = collection.insert_one({model_uuid: pickled_model, 'name': model_uuid, 'created_time': time.time()})
    print(info.inserted_id, ' saved with this id successfully!')

    details = {
        'inserted_id': info.inserted_id,
        'model_name': model_uuid,
        'created_time': time.time()
    }
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA {
      <http://www.semanticweb.org/kidz/festo#%s> festo:type 
      <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model>.}"""
    execute_sparql_query_write(query_template % model_uuid)

    store_model_kg(model_uuid, training_data_uuid)

    connect_model_to_training_run(model_uuid, "TR1")

    return model_uuid
