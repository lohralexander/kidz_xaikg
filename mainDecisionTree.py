import uuid

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree

from config import Config
from functions import (
    save_model, connect_model_to_training_run,
    create_global_insight, save_nodes_to_graph, create_global_explanation_run, create_parameter,
    create_edge_attributes, calculate_information_gain,
    calculate_entropy, create_assumption
)

# Run ID
# Prefix
run_id = str(uuid.uuid1())

# Daten einlesen
df = pd.read_csv("sparqlResult.csv")

# Datenbereinigung
df.drop('experiment', axis='columns', inplace=True)
df.drop('cylinder', axis='columns', inplace=True)

# Aufteilung der Daten
X = df.drop('result', axis=1)
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Entscheidungsbaum erstellen
# Einstellungen
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, y_train)

# Accuracy und Kappa ausrechnen
y_pred = clf.predict(X_test)
accuracy = (y_pred == y_test).mean()
kappa = cohen_kappa_score(y_test, y_pred)

# Weitere Statistiken für Global Insight
feature_importance = clf.feature_importances_
most_important_feature_index = feature_importance.argmax()
most_important_feature_name = X.columns[most_important_feature_index]

tree_num_nodes = clf.tree_.node_count
tree_depth = clf.tree_.max_depth


def visitor(node_id, parent_id, dt, decision_nodes, node, depth, parent_split_type=None):
    # ID
    global_insight_run = run_id
    enriched_node_id = f"GINS{global_insight_run}Node{node_id}"
    enriched_parent_id = f"GINS{global_insight_run}Node{parent_id}"

    # Benötigte Attribute
    feature_name = X.columns[dt.feature[node]] if dt.feature[node] != _tree.TREE_UNDEFINED else None
    threshold = round(dt.threshold[node], 2) if dt.feature[node] != _tree.TREE_UNDEFINED else None
    value = dt.value[node][0]
    predicted_class = value.argmax()
    predicted_class_name = y.unique()[predicted_class]
    sample_count = value.sum()
    value_true = value[0]
    value_false = value[1]
    accuracy = value[predicted_class] / sample_count if sample_count > 0 else 0

    # Zuweisung des Splits
    if parent_split_type == "left":
        comparison_operator = "<="
    else:
        if parent_split_type == "right":
            comparison_operator = ">"

    if parent_id is None:
        comparison_operator = "start_node"

    # Berechnung Information Gain
    total_samples = value_true + value_false

    if total_samples == 0:
        return 0  # Vermeide Division durch Null

    probability_true = value_true / total_samples
    probability_false = value_false / total_samples

    entropy_current_node = calculate_entropy(np.array([probability_true, probability_false]))

    decision_node = {
        "node_id": enriched_node_id,
        "parent_id": enriched_parent_id,
        "value": value,
        "predicted_class": predicted_class,
        "sample_count": sample_count,
        "value_true": value_true,
        "value_false": value_false,
        "accuracy": accuracy,
        "predicted_class_name": predicted_class_name,
        "feature_name": feature_name,
        "threshold": threshold,
        "comparison_operator": comparison_operator,
        "entropy_current": entropy_current_node,
    }

    decision_nodes.append(decision_node)

    if dt.feature[node] != _tree.TREE_UNDEFINED:
        left_child_id = len(decision_nodes)
        visitor(left_child_id, node_id, dt, decision_nodes, dt.children_left[node], depth + 1, parent_split_type="left")
        right_child_id = len(decision_nodes)
        visitor(right_child_id, node_id, dt, decision_nodes, dt.children_right[node], depth + 1,
                parent_split_type="right")


# Speichern der wichtigen Informationen durch die Visitor Funktion:
decision_nodes = []
visitor(0, None, clf.tree_, decision_nodes, 0, 1)

# information gain
calculate_information_gain(decision_nodes, run_id)

# assumption
create_assumption(decision_nodes, run_id)

# Versteckt ist ein Print der Informationen zu den Knoten in Sparql Form
# Beispielausgabe in Sparql Form
# for i, node in enumerate(decision_nodes):
#     print(f"<festo:{node['node_id']}> rdf:type festo:Node ;\n"
#                  f"     festo:nodeID {node['node_id']};\n"
#                  f"     festo:parentNodeID {node['parent_id']} ;\n"
#                  f"     festo:True {node['value_true']:.0f} ;\n"
#                  f"     festo:False {node['value_false']:.0f} ;\n"
#                  f"     festo:Class {node['predicted_class_name']} ;\n"
#                  f"     festo:sampleCount {node['sample_count']:.0f} ;\n"
#                  f"     festo:accuracy {node['accuracy']*100:.2f}% ;\n"
#                  f"     festo:feature {node['feature_name']} ;\n"
#                  f"     festo:threshold {node['threshold']} ;\n"
#                  f"     festo:entropy current {node['entropy_current']} ;\n"
#                  f"     festo:information_gain {node['information_gain']} ;\n"
#                  f"     festo:assumption {node['assumption']} ;\n"
#                  f"     festo:splitType {node['comparison_operator']} .\n\n"
#                  f"<festo:> {node['parent_id']} festo:hasChildNode <festo:{node['node_id']}>\n\n")


# Ab hier werden die Nodes in den KG geladen ________________________________________________________________________
# Repository wird festgelegt
repository = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3"
repository_update = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3/statements"

# Speichern der Knoten und Erstellen der Kanten:
save_nodes_to_graph(Config.graphddb_repository_update, decision_nodes, run_id)
print("Erfolgreich die Knoten geladen")

# Modell wird gespeichert
model_id = save_model(Config.graphddb_repository_update, clf, run_id)
connect_model_to_training_run(Config.graphddb_repository_update, model_id, "TR" + run_id)
print("Erfolgreich das Modell gespeichert")

# Global Explanation Run und Global Insight einfügen
create_global_explanation_run(Config.graphddb_repository_update, "GER" + run_id, run_id,
                              "Model-DecisionTreeClassifier-" + run_id)
create_global_insight(Config.graphddb_repository_update, "GINS" + run_id, run_id, accuracy, kappa, most_important_feature_name,
                      tree_depth, tree_num_nodes)
print("Erfolgreich GER und GINS erstellt und verbunden")

# Parameter einfügen
create_parameter(Config.graphddb_repository_update, clf, run_id, "Model-DecisionTreeClassifier-" + run_id)
print("Erfolgreich Parameter erstellt und verbunden")

# Verbindung zwischen Nodes und weight und pressure
create_edge_attributes(Config.graphddb_repository_update, decision_nodes)
print('Erfolgreich Nodes mit Features verbunden')
