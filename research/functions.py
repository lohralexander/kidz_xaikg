import re
import statistics

import owl
from config import Logger
from research.gptConnector import gpt_request
from research.questionnaire import Questionnaire

logger = Logger.setup_logging()


def rag(ontology: owl.Ontology, question: str, sleep_time=0):
    if not isinstance(ontology, owl.Ontology):
        Exception(TypeError, "The given ontology is not an instance of the Ontology class.")
    ontology_structure = ontology.get_ontology_structure()
    ontology_node_overview = ontology.get_ontology_node_overview()
    system = f"Here is an Ontology:{ontology_structure}. Here is an overview of the nodes:{ontology_node_overview}"
    user = (
        f"Use the given Ontology and the List of Nodes. Whith which node informations could you answer the following question? {question}. Look if you find matching Node IDs in the question and use them. Look for usable links to other nodes. Give a list of the nodes you found, no explanation.")

    found_node_id_list = re.findall(r'\w+', gpt_request(system, user, sleep_time))
    found_nodes_list = []
    for node_id in found_node_id_list:
        if node_id in ontology.node_dict:
            found_nodes_list.append(ontology.get_node(node_id))

    logger.info(f"Found nodes: {found_nodes_list}")
    retrieved_relevant_information = [str(obj) for obj in found_nodes_list]
    logger.info(retrieved_relevant_information)
    return gpt_request(f"Here is information relevant for the Question: {retrieved_relevant_information}",
                       question, sleep_time)


def work_through_the_questionnaire(ontology: owl.Ontology, questionnaire: Questionnaire):
    gpt_answers = {}
    for index, question in enumerate(questionnaire.get_questions().values(), start=1):
        gpt_answers.update({index: rag(ontology, question, 10)})
    return gpt_answers


def compare_question_answer_pair(gpt_answer, correct_answer):
    system = ("Compare the given answer with the gold standard answer. Rate them with a score between 0 (No factual match) and 5 with 5 "
              "is a perfect match. Only the facts are relevant, not the wording. Answer only with the score, "
              "nothing else.")
    user = f"The gold standard answer is {correct_answer}. The answer to score is {gpt_answer}"
    return gpt_request(system, user, sleep_time=0)


def calculate_quality_measures(score_list):
    score_numeric_list = []
    for item in score_list:
        try:
            score_numeric_list.append(int(item))
        except ValueError:
            logger.warn(f"Warning: '{item}' cannot be converted to an integer and will be ignored.")

    if not score_numeric_list:
        return "No valid numbers to process."

    mean_value = statistics.mean(score_numeric_list)
    min_value = min(score_numeric_list)
    max_value = max(score_numeric_list)
    median_value = statistics.median(score_numeric_list)

    return {
        "mean": mean_value,
        "min": min_value,
        "max": max_value,
        "median": median_value
    }

def start_research_run(ontology: owl.Ontology, questionnaire: Questionnaire):
    gpt_answers = work_through_the_questionnaire(ontology, questionnaire)
    correct_answers = questionnaire.get_answers()

    score_list = []
    for gpt_answer, correct_answer in zip(gpt_answers.values(), correct_answers.values()):
        logger.info(f"Comparing answers: /n Correct: {correct_answer} /n GPT: {gpt_answer}")
        score_list.append(compare_question_answer_pair(gpt_answer, correct_answer))

    measures = calculate_quality_measures(score_list)
    return measures
