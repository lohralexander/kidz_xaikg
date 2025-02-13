import hashlib
import statistics

from ldrag.rag import *
from research import owl
from research.questionnaire import Questionnaire


def execute_query(query, ontology):
    pattern = r"\['?([^'\]]+)'?\]"
    # pattern = "\[([A - Za - z0 - 9._] * (?:, [A-Za-z0-9._] *){0, 2})\]"
    matches = re.findall(pattern, query)
    return list(ontology.get_nodes(matches).values())


def work_through_the_questionnaire(ontology: owl.Ontology, questionnaire: Questionnaire, run_id, ):
    gpt_answers = {}
    for index, question in enumerate(questionnaire.get_questions().values(), start=1):
        gpt_response = gpt_request(user_message=question,
                                   system_message="First give a precise and short answer and then explain it in detail",
                                   retrieved_information=information_retriever(ontology=ontology, user_query=question))
        gpt_answers.update({index: gpt_response})
    return gpt_answers


def compare_question_answer_pair(gpt_answer, correct_answer):
    system = ("Compare the given answer with the gold standard answer. Rate them with a score between 0 (No factual "
              "match) and 5 with 5 is a perfect match. Only the facts are relevant, not the wording. Answer only with "
              "the score, nothing else.")
    user = f"The gold standard answer is '{correct_answer}'. The answer to score is {gpt_answer}"
    return gpt_request_with_history(system, user, sleep_time=0)[0]


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


def create_result_file(questionnaire: Questionnaire, gpt_answers, score_list, quality_measures, run_id):
    file_path = f"results/results_{run_id}.txt"

    with open(file_path, 'w', encoding='utf-8') as file:
        for key, gpt_answer in gpt_answers.items():
            cleaned_gpt_answer = gpt_answer.replace("\n", "")
            file.write(f"Unique ID: {hashlib.sha256(questionnaire.get_question(key).encode()).hexdigest()[:8]}\n"
                       f"Question {key}: {questionnaire.get_question(key)}\n"
                       f"Correct Answer {key}: {questionnaire.get_answer(key)}\n"
                       f"GPT Answer {key}: {cleaned_gpt_answer}\n"
                       f"Score: {score_list[key - 1]}\n\n")
        file.write("List of Measurements: \n")
        for key, value in quality_measures.items():
            file.write(f"{key}: {value}\n")

    return file_path


def create_overview_file(path_list, quality_measures, alternate_cycles, run_id):
    file_path = f"results/overview_measurements_{run_id}.html"
    with open(file_path, 'w') as file:
        # Write the HTML header
        file.write(f"<html>\n<head>\n<title>Run Overview {run_id}</title>\n</head>\n<body>\n")

        # Add a heading for the Run ID
        file.write(f"<h1>Run: {run_id}</h1>\n")

        # Add parameters as a subsection
        file.write(f"<h2>Parameters</h2>\n")
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


def start_research_run(ontology: owl.Ontology, questionnaire: Questionnaire,
                       alternation_cycles: int = 0):
    correct_answers = questionnaire.get_answers()

    run_id = uuid.uuid1()

    if Config.graph_gen:
        ontology.create_dynamic_instance_graph(run_id)
        ontology.create_dynamic_class_graph(run_id)

    result_path_list = []
    overview_scores_list = []

    while alternation_cycles >= 0:
        alternation_cycles -= 1
        if result_path_list:
            questionnaire.alternate_questions()

        score_list = []
        gpt_answers = work_through_the_questionnaire(ontology, questionnaire, run_id)

        for gpt_answer, correct_answer in zip(gpt_answers.values(), correct_answers.values()):
            logger.info(f"Comparing answers: /n Correct: {correct_answer} /n GPT: {gpt_answer}")
            score_list.append(compare_question_answer_pair(gpt_answer, correct_answer))

        overview_scores_list.extend(score_list)
        quality_measures = calculate_quality_measures(score_list)
        result_path_list.append(create_result_file(questionnaire, gpt_answers, score_list, quality_measures, run_id))

    aggregated_quality_measures = calculate_quality_measures(overview_scores_list)
    create_overview_file(result_path_list, aggregated_quality_measures, alternation_cycles, run_id)
    logger.info(f"Research run {run_id} finished.")
    return aggregated_quality_measures
