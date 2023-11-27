from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

repository = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3"
repository_update = "http://LAB-Thinkpad:7200/repositories/KidzDecisionTreeV3/statements"

# Deine SPARQL-Abfrage
sparql_query = """
PREFIX festo: <http://www.semanticweb.org/kidz/festo#>

SELECT ?node ?property ?value
WHERE {
  ?node festo:hasChildNode ?childNode.
  ?node ?property ?value.

  FILTER(CONTAINS(STR(?node), "GINS09Node"))
}
"""

# SPARQL-Abfrage ausführen
sparql = SPARQLWrapper(repository)
sparql.setQuery(sparql_query)
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

# Ergebnisse in einen Pandas DataFrame umwandeln
data = []
for result in results['results']['bindings']:
    node = result['node']['value'].replace('http://www.semanticweb.org/kidz/festo#', '')
    property = result['property']['value'].replace('http://www.semanticweb.org/kidz/festo#', '')
    value = result['value']['value'].replace('http://www.semanticweb.org/kidz/festo#', '')
    data.append({'node': node, 'property': property, 'value': value})

df = pd.DataFrame(data)

# DataFrame anzeigen
# print(df.head)

df_cleaned = df.drop_duplicates()

# DataFrame in TXT exportieren
df_cleaned.to_csv('DecTreeTripleResult.txt', sep='\t', index=False)
df_cleaned.to_csv('DecTreeTripleResult.csv', index=False)

