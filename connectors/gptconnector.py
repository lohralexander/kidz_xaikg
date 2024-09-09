import os

import openai
import pandas as pd

from config import Config

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.graphs.networkx_graph import KnowledgeTriple
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
import networkx as nx
import matplotlib.pyplot as plt

import requests
import json


def get_gpt_response(input: str):
    llm = ChatOpenAI(openai_api_key=os.getenv("gptkidz"))

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain.invoke({"input": input})

def get_gpt_response_with_csv_input(input: str, csv_path: str):
    csv_file_path = r'C:\Users\Alex\bwSyncShare\KIDZ\AP3 eXplainable AI\use_cases\Viergelenk\run_0_rangePM20.csv'
    df = pd.read_csv(csv_file_path)

    # CSV-Daten in einen string konvertieren
    csv_data = df.to_csv(index=False)

    # Funktion zur Kommunikation mit der OpenAI ChatGPT-API
    def query_chatgpt(prompt):
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()

    # Beispielhafte Anfrage an die API, die die CSV-Daten verwendet
    prompt = f"Hier sind die Daten aus meiner CSV-Datei:\n{csv_data}\n\nWas kann ich mit diesen Daten machen?"

    # API-Aufruf
    response_text = query_chatgpt(prompt)
    print(response_text)

def get_retrieval_gpt_response(repository_id:str, question: str, top_k: int = 10):
    url = Config.graphdb_connector + "?repositoryID=" + repository_id
    payload = json.dumps({
        "question": question,
        "askSettings": {
            "queryTemplate": {
                "query": "string"
            },
            "groundTruths": [],
            "echoVectorQuery": False,
            "topK": top_k
        },
        "history": []
    })
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)


if __name__ == '__main__':
    get_gpt_response_with_csv_input()

    #get_retrieval_gpt_response("starwars","What is the name of the character that is the father of Luke Skywalker?", top_k=10)

    # openai.api_key = os.getenv("gptkidz")
    #
    #
    # kg = [
    #     ("Prediction1", "hasInput", "Model-DecsionTreeClassifier1"),
    #     ("LocalExplanationRun1", "hasInput", "Prediction1"),
    #     ("LocalExplanationRun1", "hasOutput", "LocalInsight1"),
    #     ("LocalExplanationRun1", "hasOutput", "LocalInsight2"),
    #     ("LocalInsight1", "hasPressure", "0.5"),
    #     ("LocalInsight1", "hasPressureShapley", "-0.78"),
    #     ("LocalInsight2", "hasWeight", "50.5"),
    #     ("LocalInsight2", "hasWeightShapley", "0.109")
    # ]
    #
    # index_creator = GraphIndexCreator(llm=ChatOpenAI(openai_api_key=os.getenv("gptkidz")))
    #
    # graph = index_creator.from_text('')
    # for (node1, relation, node2) in kg:
    #     graph.add_triple(KnowledgeTriple(node1, relation, node2))
    #
    # G = nx.DiGraph()
    # for node1, relation, node2 in kg:
    #     G.add_edge(node1, node2, label=relation)
    #
    # plt.figure(figsize=(25, 25), dpi=300)
    # pos = nx.spring_layout(G, k=2, iterations=50, seed=0)
    #
    # nx.draw_networkx_nodes(G, pos, node_size=1000)
    # nx.draw_networkx_edges(G, pos, edge_color='gray', edgelist=G.edges(), width=2)
    # nx.draw_networkx_labels(G, pos, font_size=8)
    # edge_labels = nx.get_edge_attributes(G, 'label')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    #
    # plt.axis('off')
    # plt.show()
    #
    # chain = GraphQAChain.from_llm(ChatOpenAI(openai_api_key=os.getenv("gptkidz")), graph=graph, verbose=True)
    # print(chain.invoke("What do you know about Prediction1 and possible insights?"))
