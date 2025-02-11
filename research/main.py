from functions import *
from questionnaire import *
from research_config import Initialization

if __name__ == '__main__':
    research_config = Initialization(demo_mode=True)
    owl = research_config.get_ontology()
    start_research_run(research_config.owl, research_config.questionnaire, alternation_cycles=0)
