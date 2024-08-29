import re

import owl
from config import Logger
from research.gptConnector import gpt_request

logger = Logger.setup_logging()


def find_matching_objects(toCheck, object_dict):
    # Define the regular expression pattern
    pattern = r'\b(model_[a-f0-9]{8}|training_run_[a-f0-9]{8}|dataset_[a-f0-9]{8}|preprocessing_[a-f0-9]{8}|feature_[a-f0-9]{8})\b'
    # Find all matches based on the regular expression
    matches = re.findall(pattern, toCheck)
    # Look up each match in the dictionary and return the corresponding objects
    matched_objects = [object_dict[match] for match in matches if match in object_dict]
    return matched_objects


def rag(ontology: owl.Ontology, question: str):
    if not isinstance(ontology, owl.Ontology):
        Exception(TypeError, "The given ontology is not an instance of the Ontology class.")
    ontology_structure = ontology.get_ontology_structure()
    ontology_node_overview = ontology.get_ontology_node_overview()
    system = f"Here is an Ontology:{ontology_structure}. Here is an overview of the nodes:{ontology_node_overview}"
    user = (f"Use the given Ontology and the List of Nodes. Whith which node informations could you answer the following question? {question}. Look if you find matching Node IDs in the question and use them. Look for usable links to other nodes. Give a list of the nodes you found, no explanation.")

    found_node_id_list = re.findall(r'\w+', gpt_request(system, user, sleep_time=0))
    found_nodes_list = []
    for node_id in found_node_id_list:
        if node_id in ontology.node_dict:
            found_nodes_list.append(ontology.get_node(node_id))

    logger.info(f"Found nodes: {found_nodes_list}")
    retrieved_relevant_information = [str(obj) for obj in found_nodes_list]
    logger.info(retrieved_relevant_information)
    return gpt_request(f"Here is information relevant for the Question: {retrieved_relevant_information}",
                       question, sleep_time=0)
