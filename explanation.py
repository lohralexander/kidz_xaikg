import logging
import uuid

import pandas as pd
import shap
import sklearn

import dbconnector
from config import Config


def create_explanation(prediction_uuid: str):
    local_explanation_run_uuid = "LocalExplanationRun-" + str(uuid.uuid1())
    global_explanation_run_unique_identifier = "GlobalExplanationRun-" + str(uuid.uuid1())
    logging.info(local_explanation_run_uuid)
    logging.info(global_explanation_run_unique_identifier)

    training_data = dbconnector.load_saved_object_from_db(training_data_uuid, client=Config.mongodb_client,
                                                          db=Config.mongodb_database, dbconnection='data')
    training_data = training_data.drop("result", axis=1)

    combinedFrame = pd.concat([training_data, pd.DataFrame(row)], ignore_index=True)
    explainer = shap.Explainer(clf.predict, combinedFrame)
    shap_values = explainer(combinedFrame)

    for feature_name, shap_value, feature_value in zip(
            shap_values.feature_names, shap_values[-1].values, shap_values[-1].data
    ):
        local_insight_uuid = "LocalInsight-" + str(uuid.uuid1())
        query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:%s "%s";
        festo:%s_Shap "%s".}"""
        dbconnector.execute_sparql_query(
            query_template
            % (
                local_insight_uuid,
                feature_name,
                feature_value,
                feature_name,
                str(shap_value),
            )
        )

        query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:hasInput <http://www.semanticweb.org/kidz/festo#%s>;
        festo:hasOutput <http://www.semanticweb.org/kidz/festo#%s>;
        festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#LocalExplanationRun>.}"""
        dbconnector.execute_sparql_query(
            query_template
            % (local_explanation_run_uuid, prediction_uuid, local_insight_uuid)
        )

    return shap_values, shap_values[-1]
