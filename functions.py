
# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import _tree


import os
import pickle
import uuid
from urllib.parse import quote
import numpy as np
from SPARQLWrapper import JSON, SPARQLWrapper, POST


# FunktionsID: 1
# Name: visitor
# Beschreibung: Die Funktion geht den Baum entlang und speichert zu jedem Knoten wichtige Informationen.




# FunktionsID: 2
# Name: create_edge
# Beschreibung:  Hier werden die Kanten von den Knoten hinzugefügt

def create_edge(repository_update,parent_id, child_id):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
    INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:hasChildNode <http://www.semanticweb.org/kidz/festo#%s> .
    }"""

    query_string = query_template % (parent_id, child_id)

    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()

# FunktionsID: 3
# Name: save_node_to_graph
# Beschreibung: Hier werden die Knoten des Entscheidungsbaumes im Graphen gespeichert

def save_nodes_to_graph(repository_update, decision_nodes,run_id):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
    INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> rdf:type festo:Node ;
             festo:nodeID "%s" ;
             festo:parentNodeID "%s" ;
             festo:True "%s" ;
             festo:False "%s" ;
             festo:Class "%s" ;
             festo:sampleCount "%s" ;
             festo:accuracy "%s" ;
             festo:feature "%s" ;
             festo:threshold "%s" ;
             festo:splitType "%s" .
    }"""

    for node in decision_nodes:
        #unique_identifier = "GINS03Node" + str(uuid.uuid1())  # Generiere eine eindeutige Kennung
        query_string = query_template % (
            node['node_id'], node['node_id'], node['parent_id'], node['value_true'], node['value_false'],
            node['predicted_class_name'], node['sample_count'], node['accuracy'], node['feature_name'],
            node['threshold'], node['comparison_operator']
        )

        sparql.setQuery(query_string)
        sparql.method = "POST"
        sparql.query()

        # Füge die Kante für alle Knoten außer dem Wurzelknoten hinzu
        if node['parent_id'] != "GINS"+run_id+"NodeNone":
            create_edge(repository_update, node['parent_id'], node['node_id'])
            print("Edge created successfully.")

    return "Nodes uploaded successfully."

# FunktionsID: 4
# Name: save_model
# Beschreibung: Hier wird der Entscheidungsbaum an sich gespeichert.

def save_model(repository_update, classifier, run_id):
    algorithmType = type(classifier).__name__
    unique_identifier = "Model-" + algorithmType + "-" + run_id #str(uuid.uuid1())
    print(unique_identifier)
    path = os.path.abspath(".") + unique_identifier + ".pickle"
    print(quote(path))

    # SPARQL-Update mit zusätzlicher Kante "type" zu "NamedIndividual"
    sparql = SPARQLWrapper(repository_update)
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> 
                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        PREFIX kidz: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>

                        INSERT DATA 
                        {
                            <http://www.semanticweb.org/kidz/festo#%s> festo:path "%s" ;
                                festo:type <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#Model> ;
                                rdf:type owl:NamedIndividual, kidz:Model.
                        }"""

    query_string = query_template % (unique_identifier, quote(path))

    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()
    return unique_identifier


# FunktionsID: 5
# Name: connect_model_to_training_run
# Beschreibung: Hier wird das Modell einem Trainingsrun zugewiesen

def connect_model_to_training_run(repository_update, unique_identifier, training_run):
    sparql = SPARQLWrapper(repository_update)
    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#> 
                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                        PREFIX kidz: <http://www.semanticweb.org/alexa/ontologies/2023/6/kidzarchitecture#>

                        INSERT DATA {
    <http://www.semanticweb.org/kidz/festo#%s> festo:hasOutput <http://www.semanticweb.org/kidz/festo#%s> ;
                                               rdf:type owl:NamedIndividual, kidz:TrainingRun .
}"""

    query_string = query_template % (training_run, unique_identifier)

    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()

# FunktionsID: 6
# Name: create_meta_edge
# Beschreibung: Hier wird ein universelle Kante erstellt.

def create_meta_edge(repository_update, head, connection_type, child):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
    INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> festo:%s <http://www.semanticweb.org/kidz/festo#%s> .
    }"""

    query_string = query_template % (head, connection_type, child)

    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()


# FunktionsID: 7
# Name: create_global_explanation_run
# Beschreibung: Erstellt den Knoten Global Explanation Run und direkt noch einen Kante zu Global Insight
def create_global_explanation_run(repository_update, extended_run_id, run_id, unique_identifier):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
    INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> rdf:type festo:GlobalExplanationRun ;
                                                    rdf:type owl:NamedIndividual .
    }"""

    query_string = query_template % (extended_run_id)
    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()

    create_meta_edge(repository_update, "GER" + run_id, "hasOutput", "GINS" + run_id)
    create_meta_edge(repository_update, "GER" + run_id, "hasInput", unique_identifier)


# FunktionsID: 8
# Name: create_global_insight
# Beschreibung: Erstellt den Knoten Global Insight

def create_global_insight(repository_update, unique_identifier, run_id):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
    INSERT DATA {
        <http://www.semanticweb.org/kidz/festo#%s> rdf:type festo:GlobalInsight ;
                                                    rdf:type owl:NamedIndividual .
    }"""

    query_string = query_template % (unique_identifier)
    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()

    create_edge(repository_update, "GINS" + run_id, "GINS" + run_id + "Node0")

# FunktionsID: 9
# Name: create_parameter
# Beschreibung: Fügt die Parameter einem neune Knoten hinzu

def create_parameter(repository_update, clf, run_id, unique_identifier):
    sparql = SPARQLWrapper(repository_update)

    query_template = """PREFIX festo: <http://www.semanticweb.org/kidz/festo#>
            PREFIX ont: <http://www.co-ode.org/ontologies/ont.owl#>

            INSERT DATA {
                <http://www.semanticweb.org/kidz/festo#%s> rdf:type ont:Parameter ;
                 festo:max_depth "%s" .
            }"""

    query_string = query_template % ("Parameter"+run_id, clf.get_params()['max_depth'],)

    sparql.setQuery(query_string)
    sparql.method = "POST"
    sparql.query()

    create_meta_edge(repository_update, unique_identifier, "hasInput", "Parameter" + run_id)

# FunktionsID: 10
# Name: create_edge_attributes
# Beschreibung: Fügt die Kanten zwischen den einzelnen Nodes und den features ein.

def create_edge_attributes(repository_update, decision_nodes):
    sparql = SPARQLWrapper(repository_update)

    for node in decision_nodes:
        node_id = node['node_id']
        feature_name = node['feature_name']

        if feature_name == 'pressure':
            create_meta_edge(repository_update, node_id, "isUsedForSplit", "pressure")
        elif feature_name == 'weight':
            create_meta_edge(repository_update, node_id, "isUsedForSplit", "weight")

