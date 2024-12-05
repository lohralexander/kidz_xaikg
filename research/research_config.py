from research.owl import *
from research.questionnaire import Questionnaire


class Initialization():
    def __init__(self, demo_mode=False):
        self.owl = Ontology()
        #self.owl.create_demo_ontology()
        self.questionnaire = Questionnaire(demo_mode)

    def get_questionnaire(self):
        return self.questionnaire

    def get_ontology(self):
        return self.owl
