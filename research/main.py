from owl import *
from functions import *
from questionnaire import *
from research_config import *

if __name__ == '__main__':
    config = Initialization()
    owl = config.get_ontology()

    owl.create_dynamic_instance_graph()
    owl.create_dynamic_class_graph()
    questionnaire = Questionnaire(demo_mode=True)

    start_research_run(owl, questionnaire, search_depth=0, alternation_cycles=0)
