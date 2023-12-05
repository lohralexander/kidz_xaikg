import re

import pandas as pd


def extract_features(dataframe: pd.DataFrame):
    for feature in [*dataframe]:
        print(feature)


def extract_id_from_uri(full_uri: str):
    pattern = r'.*#(.*)$'

    match = re.search(pattern, full_uri)

    if match:
        return match.group(1)
