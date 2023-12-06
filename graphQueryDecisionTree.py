from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

from config import Config

#repository = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3"
#repository_update = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3/statements"

# Abfrage bisher nur mit übergebener Modell Nummer

# Deine SPARQL-Abfrage
query_template = """
                PREFIX festo: <http://www.semanticweb.org/kidz/festo#>

                SELECT ?node ?property ?value
                WHERE {
                    ?node ?property ?value.
                    OPTIONAL {?node festo:hasChildNode ?childNode.}

                FILTER(CONTAINS(STR(?node), "%s"))
                }
                """

query_string = query_template % ("GINS43988de9-8e9d-11ee-a864-3003c86b7bf0"+"Node")

# SPARQL-Abfrage ausführen
sparql = SPARQLWrapper(Config.repository)
sparql.setQuery(query_string)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Ergebnisse in einen Pandas DataFrame umwandeln
data = []
for result in results['results']['bindings']:
    node = result['node']['value'].replace('http://www.semanticweb.org/kidz/festo#GINS43988de9-8e9d-11ee-a864-3003c86b7bf0', '')
    property = result['property']['value'].replace('http://www.semanticweb.org/kidz/festo#', '').replace('http://www.w3.org/1999/02/22-rdf-syntax-ns#', '')
    value = result['value']['value'].replace('http://www.semanticweb.org/kidz/festo#', '').replace('http://www.semanticweb.org/kidz/festo#', '').replace('GINS43988de9-8e9d-11ee-a864-3003c86b7bf0', '')
    data.append({'node': node, 'property': property, 'value': value})

df = pd.DataFrame(data)


# Funktion, um die Zahl nach "Node" zu extrahieren
def extract_node_number(node_string):
    try:
        start_index = node_string.index('Node') + 4  # Index nach "Node" + Länge von "Node"
        return int(node_string[start_index:])
    except ValueError:
        return None

# Neue Spalte "node_number" erstellen
df['node_number'] = df['node'].apply(extract_node_number)

df.sort_values(by='node_number', inplace=True)

# node_number wird entfernt
df_sorted = df.drop(columns=['node_number'])

# Duplikate werden entfernt
df_sorted = df_sorted.drop_duplicates()


# DataFrame exportieren
# df_cleaned.to_csv('DecTreeTripleResult.txt', sep='\t', index=False)
df_sorted.to_csv('DecTreeTripleResult.csv', index=False)

