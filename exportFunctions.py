import json

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def sklearn_model_to_ontology(model, model_id, dataset_id, task_id, attributes, X_test, y_test, output_file):
    """
    Converts a trained sklearn model into the ontology structure and appends it to an existing JSON file.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param dataset_id: Identifier of the dataset used for training
    :param task_id: Identifier of the task the model achieves
    :param attributes: List of attribute node_ids used in training
    :param X_test: Test data features
    :param y_test: True labels for evaluation
    :param output_file: Path to the JSON file to append the ontology entry
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else np.zeros(
        (len(y_pred), model.n_classes_)) if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None).tolist()
    recall = recall_score(y_test, y_pred, average=None).tolist()
    f1 = f1_score(y_test, y_pred, average=None).tolist()
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Algorithm Name
    algorithm_name = type(model).__name__

    # Ontology Structure
    ontology_entry = {
        "node_id": model_id,
        "node_class": "Model",
        "training_information": "Trained using sklearn in Python. A split validation was used.",
        "connections": [
            [dataset_id, task_id] + attributes,
            ["trainedWith", "achieves"] + ["used"] * len(attributes)
        ],
        "algorithm": algorithm_name,
        "accuracy": accuracy,
        "precision": {f"Class {i}": p for i, p in enumerate(precision)},
        "recall": {f"Class {i}": r for i, r in enumerate(recall)},
        "f1Score": {f"Class {i}": f for i, f in enumerate(f1)},
        "confusionMatrix": conf_matrix,
    }

    if roc_auc is not None:
        ontology_entry["rocAucScore"] = roc_auc

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new model entry
    data["node_instances"].append(ontology_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Ontology model appended to {output_file}")

    return ontology_entry


import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def sklearn_model_to_ontology(model, model_id, dataset_id, task_id, attributes, X_test, y_test, output_file):
    """
    Converts a trained sklearn model into the ontology structure and appends it to an existing JSON file.

    :param model: Trained sklearn model
    :param model_id: Unique model identifier
    :param dataset_id: Identifier of the dataset used for training
    :param task_id: Identifier of the task the model achieves
    :param attributes: List of attribute node_ids used in training
    :param X_test: Test data features
    :param y_test: True labels for evaluation
    :param output_file: Path to the JSON file to append the ontology entry
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else np.zeros(
        (len(y_pred), model.n_classes_))

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=None).tolist()
    recall = recall_score(y_test, y_pred, average=None).tolist()
    f1 = f1_score(y_test, y_pred, average=None).tolist()
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Algorithm Name
    algorithm_name = type(model).__name__

    # Ontology Structure
    ontology_entry = {
        "node_id": model_id,
        "node_class": "Model",
        "training_information": "Trained using sklearn in Python. A split validation was used.",
        "connections": [
            [dataset_id, task_id] + attributes,
            ["trainedWith", "achieves"] + ["used"] * len(attributes)
        ],
        "algorithm": algorithm_name,
        "accuracy": accuracy,
        "precision": {f"Class {i}": p for i, p in enumerate(precision)},
        "recall": {f"Class {i}": r for i, r in enumerate(recall)},
        "f1Score": {f"Class {i}": f for i, f in enumerate(f1)},
        "confusionMatrix": conf_matrix,
    }

    if roc_auc is not None:
        ontology_entry["rocAucScore"] = roc_auc

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new model entry
    data["node_instances"].append(ontology_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Ontology model appended to {output_file}")

    return ontology_entry


def add_dataset_metadata_from_dataframe(dataset_id, df, domain, location, date, models, output_file):
    """
    Extracts dataset metadata from a pandas DataFrame and appends it to the ontology JSON file.

    :param dataset_id: Unique identifier for the dataset
    :param df: Pandas DataFrame containing the dataset
    :param domain: Domain of the dataset
    :param location: Location where data was recorded
    :param date: Date of data recording
    :param models: List of model node_ids that used this dataset
    :param output_file: Path to the JSON file to append the dataset entry
    """
    attributes = df.columns.tolist()
    amount_of_rows = df.shape[0]
    amount_of_attributes = df.shape[1]

    dataset_entry = {
        "node_id": dataset_id,
        "amountOfRows": amount_of_rows,
        "amountOfAttributes": amount_of_attributes,
        "node_class": "Dataset",
        "domain": domain,
        "locationOfDataRecording": location,
        "dateOfRecording": date,
        "connections": [
            attributes + models,
            ["has"] * len(attributes) + ["usedBy"] * len(models)
        ]
    }

    # Load existing JSON file
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"node_instances": []}

    # Append new dataset entry
    data["node_instances"].append(dataset_entry)

    # Save back to JSON
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Dataset metadata appended to {output_file}")

    return dataset_entry


if __name__ == '__main__':
    # Load dataset
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(df, data.target, test_size=0.2, random_state=42)

    # Train model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # Define metadata for ontology
    model_id = "model_iris_dt"
    dataset_id = "iris_dataset"
    task_id = "classification_task"
    output_file = "./research/ontology/ontologyTest.json"

    # Add dataset to ontology
    add_dataset_metadata_from_dataframe(dataset_id, df, "Iris Dataset", "Unknown Location", "2024", [model_id],
                                        output_file)

    # Add model to ontology
    sklearn_model_to_ontology(model, model_id, dataset_id, task_id, df.columns.tolist(), X_test, y_test, output_file)
