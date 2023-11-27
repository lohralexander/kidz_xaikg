import logging
import uuid

import pandas as pd

import dbconnector
from config import Config


def make_prediction(feature_values: pd.DataFrame, model_uuid: str):
    model = dbconnector.load_saved_object_from_db(model_uuid, client=Config.mongodb_client, db=Config.mongodb_database,
                                                  dbconnection='models')
    prediction_uuid = "Prediction-" + str(uuid.uuid1())
    logging.info(prediction_uuid)
    result = model.predict(feature_values)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
    <http://www.semanticweb.org/kidz/festo#%s> festo:hasInput <http://www.semanticweb.org/kidz/festo#%s>;
    festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#ApplicationRun>.}"""
    dbconnector.execute_sparql_query(query_template % (prediction_uuid, str(model_uuid)))
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
        dbconnector.execute_sparql_query(query_string)
    logging.info("Created Nodes in Knowledge Graph")

    return result, prediction_uuid
