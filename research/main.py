from owl import *
from functions import *
from questionnaire import *

if __name__ == '__main__':
    owl = Ontology()
    owl.create_demo_ontology()

    owl.create_dynamic_instance_graph()
    owl.create_dynamic_class_graph()
    questionnaire = Questionnaire(False)

    print(start_research_run(owl, questionnaire, search_depth=2, alternation_cycles=1))
