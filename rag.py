import hashlib
import re

from config import Config, logger
from connectors.gptconnector import gpt_request
from research.owl import *


def information_retriever(ontology: Ontology, question: str, sleep_time=0, ):
    # if not isinstance(ontology, owl.Ontology):
    #     Exception(TypeError, "The given ontology is not an instance of the Ontology class.")
    ontology_structure = ontology.get_ontology_structure()

    # Identify the used classes, so we don't have to give gpt every single instance to pick an anker node

    system_message = f"The following structure illustrates the class level of the ontology, which will be used to answer the subsequent questions. The node classes have instances that are not listed here. :{str(ontology_structure)}."
    user_message = f"Only give as an answer a List of Classes which could be usefull in answering the given question: {question}. Use a python usable list structure."
    gpt_response, history = gpt_request(user_message=user_message, system_message=system_message, sleep_time=sleep_time)
    found_node_class_list = re.findall(r'\w+', gpt_response)

    # Identify possible starting nodes
    instance_ids = [node.get_node_id() for node in ontology.get_instances_by_class(found_node_class_list)]

    user_message = f"Use the following list of instances: {str(instance_ids)}. Which of these instances is named in the previously given question? Only use the correct ones. You can ignore spelling error or cases. Use a python usable list structure."
    gpt_response = gpt_request(user_message=user_message, previous_conversation=history, sleep_time=sleep_time)[0]
    found_node_instances_list = re.findall(r'\w+', gpt_response)
    retrieved_node_dict = ontology.get_nodes(found_node_instances_list)

    system_message = f"You are given a starting node, which is part of an ontology. Your job is to traverse the ontology to gather enough information to answer given questions. Every node is connected to other nodes. You can find the connections under  \"'Connections':\" in the form of  \"'Connections': <name of the edge> <name of the connected node>. For example  'Connections': trainedWith data_1. You can request new nodes. To do so write [name of the requested node], for example [data_1]. You can ask for more than one instance this way. For example  [data_1, data_2]. As long as you search for new information, only use this syntax, don't explain yourself. Use the exact name of the instance and don't use the edge. Your job is to gather enough information to answer given questions. To do so, traverse trough the ontology. If you think you have enough information, write \"BREAK\". this class level ontology to orientate yourself: {str(ontology_structure)} This is your starting node: {[ontology.get_node_structure(node) for node in retrieved_node_dict.values()]}"
    user_message = question
    gpt_response, history = gpt_request(user_message=user_message, system_message=system_message, sleep_time=sleep_time)

    loop_count = 0
    while loop_count < 10 and "BREAK" not in gpt_response:
        found_node_instances = execute_query(gpt_response, ontology)
        retrieved_information = []
        if found_node_instances:
            for node in found_node_instances:
                retrieved_information.append(ontology.get_node_structure(node))
                retrieved_node_dict.update({f"{node.get_node_id()}": node})
        else:
            retrieved_information = "No instance exists for that ID. You asked for a class or searched for a non existing instance."
        user_message = f"This is the result to your query: {retrieved_information}. If you need more information, use another query, otherwise write BREAK."

        if found_node_instances:
            gpt_response, history = gpt_request(user_message=user_message, previous_conversation=history,
                                                sleep_time=sleep_time)
        else:
            gpt_response = gpt_request(user_message=user_message, previous_conversation=history, sleep_time=sleep_time)[0]
        loop_count += 1

    if Config.graph_gen:
        ontology.create_rag_instance_graph(retrieved_node_dict, "rag_graph",
                                           hashlib.sha256(question.encode()).hexdigest()[:8])
    retrieved_relevant_information = [str(obj) for obj in retrieved_node_dict.values()]
    logger.info(retrieved_relevant_information)
    return retrieved_relevant_information


def execute_query(query, ontology):
    pattern = r"\['?([^'\]]+)'?\]"
    # pattern = "\[([A - Za - z0 - 9._] * (?:, [A-Za-z0-9._] *){0, 2})\]"
    matches = re.findall(pattern, query)
    return list(ontology.get_nodes(matches).values())


if __name__ == '__main__':
    owl = Ontology()
    owl.deserialize("research/ontology/ontology.json")
    information_retriever(question="How many entries does the dataset used for training model a23b have?", ontology=owl)
