import json
import re
import uuid

from flask import jsonify

from config import logger
from connectors.gptconnector import gpt_request
from research.owl import *


def information_retriever(ontology: Ontology, question: str, anker_points=None, sleep_time=0, ):
    # if not isinstance(ontology, owl.Ontology):
    #     Exception(TypeError, "The given ontology is not an instance of the Ontology class.")
    ontology_structure = ontology.get_ontology_structure()
    # if anker_points is not None:
    # Identify the used classes, so we don't have to give gpt every single instance to pick an anker node
    system_message = f"The following structure illustrates the class level of the ontology, which will be used to answer the subsequent questions. The node classes have instances that are not listed here. :{json.dumps(ontology_structure)}."
    user_message = f"Only give as an answer a List of Classes which could be useful in answering the given question: {question} Return only JSON Syntax without prefix."
    gpt_response, history = gpt_request(user_message=user_message, system_message=system_message, sleep_time=sleep_time)
    found_node_class_list = re.findall(r'\w+', gpt_response)

    # Identify possible starting nodes
    instance_ids = [node.get_node_id() for node in ontology.get_instances_by_class(found_node_class_list)]
    user_message = f"Use the following list of instances: {str(instance_ids)}. Which of these instances is named in the previously given question? Only use the correct ones. You can ignore spelling error or cases. Return only JSON Syntax."
    gpt_response = gpt_request(user_message=user_message, previous_conversation=history, sleep_time=sleep_time)[0]
    found_node_instances_list = re.findall(r'\w+', gpt_response)
    retrieved_node_dict = ontology.get_nodes(found_node_instances_list)

    system_message = f"You are given a starting node, which is part of an ontology. Your job is to traverse the ontology to gather enough information to answer given questions. Every node is connected to other nodes. You can find the connections under  \"'Connections':\" in the form of  \"'Connections': <name of the edge> <name of the connected node>. For example  'Connections': trainedWith data_1. You can request new nodes. To do so write [name of the requested node], for example [data_1]. You can ask for more than one instance this way. For example  [data_1, data_2]. As long as you search for new information, only use this syntax, don't explain yourself. Use the exact name of the instance and don't use the edge. Your job is to gather enough information to answer given questions. To do so, traverse trough the ontology. If you think you have enough information, write \"BREAK\". Use this class level ontology to orientate yourself: {str(ontology_structure)} This is your starting node: {[ontology.get_node_structure(node) for node in retrieved_node_dict.values()]}. Return only JSON Syntax without prefix."
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
            logger.debug(f"RETRIEVED INFORMATION: {retrieved_information}")
        else:
            retrieved_information = "No instance exists for that ID. You asked for a class or searched for a non existing instance."
        user_message = f"This is the result to your query: {retrieved_information}. If you need more information, use another query, otherwise write BREAK. Return only JSON Syntax without prefix."

        if found_node_instances:
            gpt_response, history = gpt_request(user_message=user_message, previous_conversation=history,
                                                sleep_time=sleep_time)
        else:
            gpt_response = gpt_request(user_message=user_message, previous_conversation=history, sleep_time=sleep_time)[
                0]
        loop_count += 1

    retrieved_graph_id = uuid.uuid1()
    graph_path = create_rag_instance_graph(retrieved_node_dict, retrieved_graph_id, question)
    retrieved_relevant_information = []
    for node in retrieved_node_dict.values():
        retrieved_relevant_information.append(ontology.get_node_structure(node))
    #retrieved_relevant_information = [str(obj) for obj in retrieved_node_dict.values()]
    logger.debug(retrieved_relevant_information)
    return retrieved_relevant_information, graph_path


def execute_query(query, ontology):
    pattern = r"\b([a-zA-Z_1-9]+)"
    #pattern = r"\['?([^'\]]+)'?\]"
    # pattern = "\[([A - Za - z0 - 9._] * (?:, [A-Za-z0-9._] *){0, 2})\]"
    matches = re.findall(pattern, query)
    return list(ontology.get_nodes(matches).values())


def create_rag_instance_graph(rag_dict, question_id, question):
    net = Network(height="100vh", width="100vw", directed=True)

    for node in rag_dict.values():
        net.add_node(node.get_node_id(), title=node.get_internal_structure())

    for node in rag_dict.values():
        for connection, edge in zip(node.get_node_connections()[0], node.get_node_connections()[1]):
            if connection in rag_dict.keys():
                net.add_edge(node.get_node_id(), connection, label=edge, arrows="to", length=400)

    output_file = f"static/graph/rag_{question_id}.html"

    directory = os.path.dirname(output_file)
    if not os.path.exists(directory):
        os.makedirs(directory)
    net.save_graph(output_file)

    with open(output_file, 'r') as file:
        html_content = file.read()

    headline_html = f"<h2 style='text-align:center;color:black;font:arial;margin-top:20px;'>Used nodes for question: {question}</h2>"
    html_content = html_content.replace('<body>', f'<body>\n{headline_html}', 1)

    with open(output_file, 'w') as file:
        file.write(html_content)

    return output_file


if __name__ == '__main__':
    owl = Ontology()
    owl.deserialize("research/ontology/ontology.json")
    information_retriever(question="How does other models perform on the task of model a23b?", ontology=owl)
