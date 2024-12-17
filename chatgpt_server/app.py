import copy

from flask import Flask, request, jsonify, render_template, session

from config import logger
from connectors.gptconnector import gpt_request
from rag import information_retriever
from research.owl import Ontology

app = Flask(__name__)
owl = Ontology()
owl.deserialize("../research/ontology/ontology.json")
app.secret_key = 'BzPopVRViW'


def rag(question):
    logger.debug(f"Conversation History: {session.get('conversation_history', None)}")
    retrieved_information, graph_path = information_retriever(ontology=owl,
                                                              user_query=question,
                                                              previous_conversation=copy.deepcopy(
                                                                  session.get('conversation_history', [])))
    logger.info(f"Retrieved information: {retrieved_information}")
    logger.debug(f"Conversation History: {session.get('conversation_history', None)}")
    gpt_response, history = gpt_request(user_message=question,
                                        previous_conversation=copy.deepcopy(session.get('conversation_history', [])),
                                        retrieved_information=retrieved_information)
    logger.info(f"History: {history}")
    session['conversation_history'] = history
    logger.info(f"Assistend response: {gpt_response}")
    return gpt_response, graph_path


@app.route("/")
def index():
    session['conversation_history'] = []
    return render_template("index.html")


@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.json
    input_text = data.get("input", "")
    logger.info(f"Incoming User request: {input_text}")
    if not input_text:
        return jsonify({"response": "Error: No input provided."}), 400
    response, graph_path = rag(input_text)
    return jsonify({"response": response,
                    "dynamicFileUrl": f"{graph_path}"})


if __name__ == "__main__":
    app.run(debug=True)
