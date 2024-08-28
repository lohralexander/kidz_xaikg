import re


def find_matching_objects(toCheck, object_dict):
    # Define the regular expression pattern
    pattern = r'\b(model_[a-f0-9]{8}|training_run_[a-f0-9]{8}|dataset_[a-f0-9]{8}|preprocessing_[a-f0-9]{8}|feature_[a-f0-9]{8})\b'
    # Find all matches based on the regular expression
    matches = re.findall(pattern, toCheck)
    # Look up each match in the dictionary and return the corresponding objects
    matched_objects = [object_dict[match] for match in matches if match in object_dict]
    return matched_objects


def rag_chatgpt(question):
    node_info = NodeInformation()
    ontology = node_info.get_node_info()
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Here is an Ontology:{ontology}."},
            {"role": "user",
             "content": f"Use the given Ontology. Which of the nodes are relevant for the following question? {question}. Give your answer in the following form only:[firstfound,secondfound,...]. If there are ID's used for the Node, add them with an underscore. Example:[model_a2f6fb37, training_run_76d864c9, dataset_58ddb600, preprocessing_b9875fe0, feature_87176016] Important: Give only the bracketed andwer, skip everything else!"}
        ]
    )

    return response.choices[0].message['content']
