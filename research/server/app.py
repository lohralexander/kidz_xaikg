from flask import Flask, request, jsonify, render_template
from research import functions
from research.owl import Ontology

app = Flask(__name__)
owl = Ontology()
owl.deserialize("../ontology/ontology.json")


def rag(input_text):
    return functions.rag_advanced(owl, input_text)


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
