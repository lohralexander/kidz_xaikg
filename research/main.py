from owl import *
from functions import *
from questionnaire import *

if __name__ == '__main__':
    owl = Ontology()
    owl.create_demo_ontology()
    connections = owl.get_connected_nodes(owl._node_dict['model_a2f6fb37'], 3)
    #print(connections)
    #print(owl.get_ontology_structure())
    #owl.create_graph()
    #print(rag(owl, "How many entries does data set DataSet_58ddb600?"))
    #print(rag(owl, "Which Dataset did model_a2f6fb37 used and how many entries does it have?"))
    questionnaire = Questionnaire()
    #print(work_through_the_questionnaire(owl, questionnaire))

    print(start_research_run(owl, questionnaire, search_depth=1, alternation_cycles=1))
