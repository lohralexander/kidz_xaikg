import logging

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import training
from explanation import create_explanation
from prediction import make_prediction

if __name__ == '__main__':
    # Set Logging Level
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)

    # Prepare Data
    df = pd.read_csv("data/sparqlResult.csv")
    df.drop("experiment", axis="columns", inplace=True)
    df.drop("cylinder", axis="columns", inplace=True)

    # Train Model
    model_uuid, training_data_uuid = training.train_model(
        DecisionTreeClassifier(random_state=0, min_samples_leaf=20, min_samples_split=40), df, "result")
    print(model_uuid)
    print(training_data_uuid)
    # Prepare Prediction Data
    row = pd.DataFrame([(0.5, 50.5)], columns=["pressure", "weight"])
    result, prediction_uuid = make_prediction(row, model_uuid)
    print(prediction_uuid)
    print(result)

    # Explain Prediction
    mean, single = create_explanation(prediction_uuid)

    print(single)