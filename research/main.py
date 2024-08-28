from owl import *

if __name__ == '__main__':
    owl = Ontology()
    owl.create_basic_ontology()
    connections = owl.get_connected_nodes(owl.node_dict['model_a2f6fb37'], 3)
    print(connections)
    print(owl.get_ontology_structure())

