import pandas as pd
import shap

import dbconnector


def create_explanation(prediction_uuid: str):
    model_uuid, training_data_uuid = dbconnector.get_prediction_information(prediction_uuid)

    training_data = dbconnector.load_saved_object_from_db(training_data_uuid, connection_id='data')
    training_data = training_data.drop("result", axis=1)

    prediction_data = dbconnector.load_saved_object_from_db(prediction_uuid, connection_id='predictions')

    model = dbconnector.load_saved_object_from_db(model_uuid, connection_id='models')

    combined_frame = pd.concat([training_data, pd.DataFrame(prediction_data)], ignore_index=True)
    explainer = shap.Explainer(model.predict, combined_frame)
    shap_values = explainer(combined_frame)

    dbconnector.write_shap_to_kg(prediction_uuid, shap_values)

    dbconnector.write_gini_importance_to_kg(prediction_uuid, model.feature_names_in_, model.feature_importances_)
    return shap_values, shap_values[-1]
