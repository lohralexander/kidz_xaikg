import logging
import pickle
import time
import uuid

import pandas as pd

from connectors import dbconnector


def make_prediction(feature_values: pd.DataFrame, model_uuid: str):
    model = dbconnector.load_saved_object_from_db(model_uuid, connection_id='models')
    prediction_uuid = "Prediction-" + str(uuid.uuid1())
    logging.info(prediction_uuid)
    result = model.predict(feature_values)

    save_prediction_data_to_db(prediction_uuid, feature_values, dbconnection='predictions')

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
    <http://www.semanticweb.org/kidz/festo#%s> festo:hasInput <http://www.semanticweb.org/kidz/festo#%s>;
    festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#ApplicationRun>.}"""
    dbconnector.execute_sparql_query_write(query_template % (prediction_uuid, str(model_uuid)))
    logging.info("Connected Nodes in Knowledge Graph")

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA 
    {<http://www.semanticweb.org/kidz/festo#%s> festo:result "%s";
    festo:%s "%s".}"""

    feature_names = [*feature_values]
    feature_values = feature_values.iloc[0].tolist()
    for feature_name, feature_value in zip(feature_names, feature_values):
        query_string = query_template % (
            prediction_uuid,
            result,
            feature_name,
            feature_value,
        )
        dbconnector.execute_sparql_query_write(query_string)
    logging.info("Created Nodes in Knowledge Graph")

    return result, prediction_uuid


def save_prediction_data_to_db(prediction_uuid: str, feature_values: pd.DataFrame, dbconnection):
    pickled_data = pickle.dumps(feature_values)
    collection = dbconnector.get_mongo_collection(dbconnection)

    info = collection.insert_one(
        {prediction_uuid: pickled_data, 'name': prediction_uuid, 'created_time': time.time()})
    logging.info(str(info.inserted_id) + ' saved successfully!')

    details = {
        'inserted_id': info.inserted_id,
        'data_name': prediction_uuid,
        'created_time': time.time()
    }

    return details
