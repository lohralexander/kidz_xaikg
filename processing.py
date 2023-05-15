import re


def get_values_from_json(x):
    return x.get('value')


def remove_owl_uri(x):
    return re.search("#(.*)", x).group(1)
