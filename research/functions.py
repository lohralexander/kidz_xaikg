import re
import statistics
import uuid

import owl
from config import logger
from research.gptConnector import gpt_request
from research.questionnaire import Questionnaire


def rag(ontology: owl.Ontology, question: str, search_depth=0, sleep_time=0, advanced_search=True):
    if not isinstance(ontology, owl.Ontology):
        Exception(TypeError, "The given ontology is not an instance of the Ontology class.")
    ontology_structure = ontology.get_ontology_structure()

    # RAG Step 1
    system = (f"Use the following Ontology:{ontology_structure}. Only give as an answer a List of Nodes which could be "
              f"usefull in answering the given question. Use a python usable list structure.")
    user = f"{question}"
    found_node_class_list = re.findall(r'\w+', gpt_request(system, user, sleep_time))

    # RAG Step 2
    instances = ontology.get_instances_by_class(found_node_class_list)
    instance_ids = []
    for node in instances:
        instance_ids.append(node.get_node_id())
    system = (
        f"Use the following list of instances: {instance_ids}. Which of these instances is named in the question. Use a python usable list structure.")
    user = f"{question}"
    found_node_instances_list = re.findall(r'\w+', gpt_request(system, user, sleep_time))
    found_nodes_dict = ontology.get_nodes(found_node_instances_list)

    logger.info(f"Found nodes: {found_nodes_dict}")

    if search_depth != 0:
        graph_search(found_nodes_dict, ontology, search_depth, advanced_search,found_node_class_list)

    instance_structure = ontology.get_node_structure(instances)
    retrieved_relevant_information = [str(obj) for obj in found_nodes_dict.values()]
    logger.info(retrieved_relevant_information)
    return gpt_request(f"Here is information relevant for the Question: {retrieved_relevant_information}",
                       question, sleep_time)


def graph_search(found_nodes_dict, ontology, search_depth, advanced_search, found_node_class_list):
    extended_depth_search_dict = found_nodes_dict.copy()
    for node in found_nodes_dict.values():
        connected_nodes_dict = ontology.get_connected_nodes(node, search_depth)
        if advanced_search:
            for key, value in list(connected_nodes_dict.items()):
                if value.get_node_class_id() not in found_node_class_list:
                    connected_nodes_dict.pop(key)
        extended_depth_search_dict.update({
            key: value for key, value in connected_nodes_dict.items()
            if key not in extended_depth_search_dict
        })
    found_nodes_dict.clear()
    found_nodes_dict.update(extended_depth_search_dict)
    extended_depth_search_dict.clear()


def work_through_the_questionnaire(ontology: owl.Ontology, questionnaire: Questionnaire, search_depth=0):
    gpt_answers = {}
    for index, question in enumerate(questionnaire.get_questions().values(), start=1):
        gpt_answers.update({index: rag(ontology, question, search_depth=search_depth, sleep_time=5)})
    return gpt_answers


def compare_question_answer_pair(gpt_answer, correct_answer):
    system = ("Compare the given answer with the gold standard answer. Rate them with a score between 0 (No factual "
              "match) and 5 with 5 is a perfect match. Only the facts are relevant, not the wording. Answer only with "
              "the score, nothing else.")
    user = f"The gold standard answer is '{correct_answer}'. The answer to score is {gpt_answer}"
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
    logger.info(f"Mean: {mean_value}")
    logger.info(f"Median: {median_value}")
    logger.info(f"Min: {min_value}")
    logger.info(f"Max: {max_value}")
    return {
        "mean": mean_value,
        "min": min_value,
        "max": max_value,
        "median": median_value
    }


def create_result_file(questionnaire: Questionnaire, gpt_answers, score_list, quality_measures):
    file_path = f"results_{uuid.uuid1()}.txt"

    with open(file_path, 'w') as file:
        for key, gpt_answer in gpt_answers.items():
            cleaned_gpt_answer = gpt_answer.replace("\n", "")
            file.write(f"Question {key}: {questionnaire.get_question(key)}\n"
                       f"Correct Answer {key}: {questionnaire.get_answer(key)}\n"
                       f"GPT Answer {key}: {cleaned_gpt_answer}\n"
                       f"Score: {score_list[key - 1]}\n\n")
        file.write("List of Measurements: \n")
        for key, value in quality_measures.items():
            file.write(f"{key}: {value}\n")

    return file_path


def create_overview_file(path_list, quality_measures, search_depth, alternate_cycles):
    run_id = uuid.uuid1()

    file_path = f"overview_measurements_{run_id}.html"
    with open(file_path, 'w') as file:
        # Write the HTML header
        file.write(f"<html>\n<head>\n<title>Run Overview {run_id}</title>\n</head>\n<body>\n")

        # Add a heading for the Run ID
        file.write(f"<h1>Run: {run_id}</h1>\n")

        # Add parameters as a subsection
        file.write(f"<h2>Parameters</h2>\n")
        file.write(f"<p><strong>Search Depth:</strong> {search_depth}</p>\n")
        file.write(f"<p><strong>Question Alternation Cycles:</strong> {alternate_cycles}</p>\n")

        # Add a list for paths
        file.write(f"<h2>Paths</h2>\n")
        file.write(f"<ol>\n")  # Ordered list for enumerating paths
        for index, path in enumerate(path_list, start=1):
            file.write(f"<li>{path}</li>\n")
        file.write(f"</ol>\n")

        # Add measurements as a subsection
        file.write(f"<h2>List of Measurements</h2>\n")
        file.write(f"<ul>\n")  # Unordered list for measurements
        for key, value in quality_measures.items():
            file.write(f"<li><strong>{key}:</strong> {value}</li>\n")
        file.write(f"</ul>\n")

        # Close the HTML tags
        file.write(f"</body>\n</html>")


def start_research_run(ontology: owl.Ontology, questionnaire: Questionnaire, search_depth: int,
                       alternation_cycles: int = 0):
    correct_answers = questionnaire.get_answers()

    result_path_list = []
    overview_scores_list = []

    while alternation_cycles >= 0:
        alternation_cycles -= 1
        if result_path_list:
            questionnaire.alternate_questions()

        score_list = []
        gpt_answers = work_through_the_questionnaire(ontology, questionnaire, search_depth=search_depth)

        for gpt_answer, correct_answer in zip(gpt_answers.values(), correct_answers.values()):
            logger.info(f"Comparing answers: /n Correct: {correct_answer} /n GPT: {gpt_answer}")
            score_list.append(compare_question_answer_pair(gpt_answer, correct_answer))

        overview_scores_list.extend(score_list)
        quality_measures = calculate_quality_measures(score_list)
        result_path_list.append(create_result_file(questionnaire, gpt_answers, score_list, quality_measures))

    aggregated_quality_measures = calculate_quality_measures(overview_scores_list)
    create_overview_file(result_path_list, aggregated_quality_measures, search_depth, alternation_cycles)
    logger.info("Research run finished.")
    return aggregated_quality_measures
