import pandas as pd
import sklearn

import dbconnector


# TODO Auslagern des Trainings und aufteilen in Abspeichern. Nutzer sollten selbst trainieren
def train_model(model: sklearn.base, training_data: pd.DataFrame, label: str):
    X = training_data.drop(label, axis=1)
    y = training_data[label]
    model = model.fit(X, y)

    training_data_uuid = dbconnector.save_training_data_to_db(training_data=training_data)
    model_uuid = dbconnector.save_model_to_db(model, training_data_uuid)
    return model_uuid, training_data_uuid


if __name__ == '__main__':
    frame = pd.read_csv("data/sparqlResult.csv")
    # extract_features(frame)
