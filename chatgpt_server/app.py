from flask import Flask, request, jsonify, render_template, session

from connectors.gptconnector import gpt_request
from rag import information_retriever
from research.owl import Ontology
from config import logger


app = Flask(__name__)
owl = Ontology()
owl.deserialize("../research/ontology/ontology.json")
app.secret_key = 'BzPopVRViW'

def rag(input_text):
    if 'retrieved_information' not in session:
        session['retrieved_information'] = []
    if 'server_history' not in session:
        session['server_history'] = []
    retrieved_information, graph_path = information_retriever(owl, input_text)
    logger.info(f"RETRIEVED INFORMATION: {retrieved_information}")
    if retrieved_information:
        session['retrieved_information'].append(retrieved_information)
        logger.debug(f"SERVER RI: {session['retrieved_information']}")
    gpt_response, history = gpt_request(user_message=input_text,
                                        previous_conversation=session.get('server_history', None),
                                        retrieved_information=session.get('retrieved_information', None))
    session['server_history'] = history
    logger.info(f"ASSISTANT RESPONSE: {gpt_response}")
    logger.info(f"HISTORY: {history}")
    return gpt_response, graph_path


@app.route("/")
def index():
    # Serve the HTML interface
    return render_template("index.html")


@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.json
    input_text = data.get("input", "")
    if not input_text:
        return jsonify({"response": "Error: No input provided."}), 400
    response, graph_path = rag(input_text)
    return jsonify({"response": response,
                    "dynamicFileUrl": f"{graph_path}"})


if __name__ == "__main__":
    app.run(debug=True)
