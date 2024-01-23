import logging
import pickle
import time
import uuid

import pandas as pd
import pymongo
from SPARQLWrapper import SPARQLWrapper, JSON

import util
from config import Config


def store_model_kg(model_uuid: str, training_data_uuid: str):
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:type 
        <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#TrainingData>;
        festo:trained <http://www.semanticweb.org/kidz/festo#%s> .}"""
    return execute_sparql_query_write(query_template % (training_data_uuid, model_uuid))


def save_training_data_to_db(training_data: pd.DataFrame):
    training_data_uuid = "TrainingData-" + str(uuid.uuid1())
    pickled_data = pickle.dumps(training_data)
    connection = get_mongo_collection('data')
    info = connection.insert_one(
        {training_data_uuid: pickled_data, 'name': training_data_uuid, 'created_time': time.time()})
    logging.info(str(info.inserted_id) + ' saved successfully!')

    # for feature in [*training_data]:

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


def get_mongo_collection(connection_id: str):
    mongo_client = pymongo.MongoClient(Config.mongodb_client)

    mongo_database = mongo_client[Config.mongodb_database]

    return mongo_database[connection_id]


def load_saved_object_from_db(object_name, connection_id):
    json_data = {}
    collection = get_mongo_collection(connection_id)
    data = collection.find({'name': object_name})

    for i in data:
        json_data = i
    pickled_model = json_data[object_name]

    return pickle.loads(pickled_model)


def load_saved_objects_from_db(object_name, connection_id: str):
    collection = get_mongo_collection(connection_id)

    query = {"name": {"$regex": object_name}}
    docs = collection.count_documents(query)

    data = collection.find({'name': {"$regex": object_name}})

    print(str(docs) + " Items found!")
    return data


def write_gini_importance_to_kg(model_uuid, gini_importance):
    local_explanation_run_uuid = "LocalExplanationRun-" + str(uuid.uuid1())
    logging.info(local_explanation_run_uuid)

    for feature_name, gini_value in zip(
            gini_importance.feature_names, gini_importance.feature_importances_
    ):
        gini_importance_uuid = "GiniImportance-" + str(uuid.uuid1())
        query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:%s "%s".}"""
        execute_sparql_query_write(
            query_template
            % (
                gini_importance_uuid,
                feature_name,
                str(gini_value),
            )
        )

        query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:hasInput <http://www.semanticweb.org/kidz/festo#%s>;
        festo:hasOutput <http://www.semanticweb.org/kidz/festo#%s>;
        festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#GiniImportanceRun>.}"""
        execute_sparql_query_write(
            query_template
            % (gini_importance_uuid, model_uuid, gini_importance_uuid)
        )


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
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        print(result["usedmodel"]["value"])
        return result["usedmodel"]["value"]


def execute_sparql_query_write(query_string: str):
    sparql = SPARQLWrapper(Config.repository_update)
    sparql.setQuery(query_string)
    sparql.method = "POST"
    query = sparql.query()
    return query


def execute_sparql_query_read(query_string: str):
    sparql = SPARQLWrapper(Config.repository)
    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def connect_model_to_training_run(model_uii, training_run):
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> INSERT DATA 
    {<http://www.semanticweb.org/kidz/festo#%s> festo:hasOutput <http://www.semanticweb.org/kidz/festo#%s>.}"""
    execute_sparql_query_write(query_template % (training_run, model_uii))


def get_prediction_information(prediction_uuid):
    global training_iri

    query_template = """ PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX kidzarchitecture: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
           SELECT ?usedmodel WHERE {<http://www.semanticweb.org/kidz/festo#%s> festo:hasInput ?usedmodel .
               ?usedmodel festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model>.}"""
    results = execute_sparql_query_read(query_template % prediction_uuid)

    model_iri = ""
    for result in results["results"]["bindings"]:
        print(result["usedmodel"]["value"])
        model_iri = result["usedmodel"]["value"]
    model_uuid = util.extract_id_from_uri(model_iri)
    query_template = """ PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX kidzarchitecture: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX : <http://www.semanticweb.org/kidz/festo>
           SELECT ?useddata WHERE {?useddata festo:trained <http://www.semanticweb.org/kidz/festo#%s>.
               ?useddata festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#TrainingData>.}"""
    results = execute_sparql_query_read(query_template % model_uuid)

    training_iri = ""
    for result in results["results"]["bindings"]:
        print(result["useddata"]["value"])
        training_iri = result["useddata"]["value"]
    training_data_uuid = util.extract_id_from_uri(training_iri)
    return model_uuid, training_data_uuid


def write_shap_to_kg(prediction_uuid, shap_values):
    local_explanation_run_uuid = "LocalExplanationRun-" + str(uuid.uuid1())
    logging.info(local_explanation_run_uuid)

    for feature_name, shap_value, feature_value in zip(
            shap_values.feature_names, shap_values[-1].values, shap_values[-1].data
    ):
        local_insight_uuid = "LocalInsight-" + str(uuid.uuid1())
        query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:%s "%s";
        festo:%s_Shap "%s".}"""
        execute_sparql_query_write(
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
        execute_sparql_query_write(
            query_template
            % (local_explanation_run_uuid, prediction_uuid, local_insight_uuid)
        )
