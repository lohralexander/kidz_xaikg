import logging
import uuid

import pandas as pd
import shap
import sklearn
from SPARQLWrapper import JSON, SPARQLWrapper

import dbconnector
from config import Config
from processing import extract_id_from_uri


def create_explanation(prediction_uuid: str):
    local_explanation_run_uuid = "LocalExplanationRun-" + str(uuid.uuid1())
    global_explanation_run_unique_identifier = "GlobalExplanationRun-" + str(uuid.uuid1())
    logging.info(local_explanation_run_uuid)
    logging.info(global_explanation_run_unique_identifier)

    query_template = """ PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX kidzarchitecture: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
           SELECT ?usedmodel WHERE {<http://www.semanticweb.org/kidz/festo#%s> festo:hasInput ?usedmodel .
               ?usedmodel festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model>.}"""

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
        model_iri = result["usedmodel"]["value"]

    model_uuid = extract_id_from_uri(model_iri)
    training_data_uuid = ""
    query_template = """ PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX kidzarchitecture: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX : <http://www.semanticweb.org/kidz/festo>
           SELECT ?useddata WHERE {?useddata festo:trained <http://www.semanticweb.org/kidz/festo#%s>.
               ?useddata festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#TrainingData>.}"""

    sparql = SPARQLWrapper(Config.repository)
    sparql.setQuery(query_template
                    % (model_uuid))
    # Setze das gewünschte Rückgabefomat (json, xml, etc.)
    sparql.setReturnFormat(JSON)

    # Führe die SPARQL-Abfrage aus und erhalte das Ergebnis
    results = sparql.query().convert()

    # Verarbeite das Ergebnis (z.B., Ausgabe der Ergebnisse)
    for result in results["results"]["bindings"]:
        print(result["useddata"]["value"])
        training_iri = result["useddata"]["value"]

    training_data_uuid = extract_id_from_uri(training_iri)
    training_data = dbconnector.load_saved_object_from_db(training_data_uuid, client=Config.mongodb_client,
                                                          db=Config.mongodb_database, dbconnection='data')
    training_data = training_data.drop("result", axis=1)

    prediction_data = dbconnector.load_saved_object_from_db(prediction_uuid, client=Config.mongodb_client,
                                                          db=Config.mongodb_database, dbconnection='predictions')

    model = dbconnector.load_saved_object_from_db(model_uuid, client=Config.mongodb_client,
                                                          db=Config.mongodb_database, dbconnection='models')

    combinedFrame = pd.concat([training_data, pd.DataFrame(prediction_data)], ignore_index=True)
    explainer = shap.Explainer(model.predict, combinedFrame)
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
