import json


def convert():
    # Load the old ontology file
    with open("ontology.json", "r", encoding="utf-8") as file:
        ontology_data = json.load(file)

    # Convert node classes
    for node_class in ontology_data["node_classes"]:
        connections = node_class.get("class_connections", [])
        if len(connections) == 2:
            node_class["class_connections"] = [
                {"target": tgt, "relation": rel} for tgt, rel in zip(connections[0], connections[1])
            ]

    # Convert node instances
    for node_instance in ontology_data["node_instances"]:
        connections = node_instance.get("connections", [])
        if len(connections) == 2:
            node_instance["connections"] = [
                {"target": tgt, "relation": rel} for tgt, rel in zip(connections[0], connections[1])
            ]

    # Save the new ontology file
    new_file_path = "ldrag/ontology_converted.json"
    with open(new_file_path, "w", encoding="utf-8") as file:
        json.dump(ontology_data, file, indent=4, ensure_ascii=False)

    print(f"Converted ontology saved to: {new_file_path}")


if __name__ == '__main__':
    convert()
