import logging

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import dbconnector
import training
from explanation import create_explanation
from prediction import make_prediction

if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)
    df = pd.read_csv("data/sparqlResult.csv")
    df.drop("experiment", axis="columns", inplace=True)
    df.drop("cylinder", axis="columns", inplace=True)
    model_uuid, training_data_uuid = training.train_model(
        DecisionTreeClassifier(random_state=0, min_samples_leaf=20, min_samples_split=40), df, "result")
    clf = dbconnector.load_saved_object_from_db(model_uuid, client='mongodb://localhost:27017/', db='datalake',
                                                dbconnection='models')

    row = pd.DataFrame([(0.5, 50.5)], columns=["pressure", "weight"])
    result, prediction_id = make_prediction(row, model_uuid)
    create_explanation(prediction_id)
