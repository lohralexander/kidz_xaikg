from flask import Flask, request, jsonify, render_template
from rag import information_retriever
from research.owl import Ontology
from connectors.gptconnector import gpt_request

app = Flask(__name__)
owl = Ontology()
owl.deserialize("../research/ontology/ontology.json")
server_history = []


def rag(input_text):
    retrieved_information = information_retriever(owl, input_text)
    gpt_response, history = gpt_request(input_text, retrieved_information=retrieved_information)
    server_history.append(history)
    return gpt_response


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
    response = rag(input_text)
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True)
