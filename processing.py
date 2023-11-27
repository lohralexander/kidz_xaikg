import re


def get_values_from_json(x):
    return x.get('value')


def remove_owl_uri(x):
    return re.search("#(.*)", x).group(1)

def extract_id_from_uri(full_uri:str):
    # Regex, um alles nach dem letzten Slash zu extrahieren
    pattern = re.compile(r'[^/]+(?=#|$)')

    # Suche nach dem Muster in der URL
    match = re.search(pattern, str(full_uri))

    # Extrahiere das Ergebnis, wenn ein Treffer vorhanden ist
    if match:
        result = match.group(0)
        print(result)
        return result
    else:
        print("Kein Treffer gefunden.")