import re


def get_values_from_json(x):
    return x.get('value')


def remove_owl_uri(x):
    return re.search("#(.*)", x).group(1)

def extract_id_from_uri(full_uri:str):
    pattern = r'.*#(.*)$'

    match = re.search(pattern, full_uri)

    if match:
        result = match.group(1)
        return result