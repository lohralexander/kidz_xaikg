import datetime
import random
import webbrowser

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network


def _insert_headline(headline, output_file):
    with open(output_file, 'r') as file:
        html_content = file.read()

    headline_html = f"<h1 style='text-align:center;color:black;font:arial;margin-top:20px;'>{headline}</h1>"
    html_content = html_content.replace('<body>', f'<body>\n{headline_html}', 1)

    with open(output_file, 'w') as file:
        file.write(html_content)


class Ontology:

    def __init__(self):
        self._node_dict = {}

    def create_demo_ontology(self, model_count=0):
        datasets = Dataset.generate_multiple_nodes(model_count)
        training_runs = TrainingRun.generate_multiple_nodes(model_count)
        models = Model.generate_multiple_nodes(model_count)
        features = Feature.generate_multiple_nodes(model_count)
        attributes = Attribute.generate_multiple_nodes(model_count)
        preprocessings = Preprocessing.generate_multiple_nodes(model_count)

        for model, dataset, training_run, feature, attribute, preprocessing in zip(models, datasets, training_runs,
                                                                                   features, attributes,
                                                                                   preprocessings):
            self._node_dict.update({model.node_id: model, dataset.node_id: dataset, training_run.node_id: training_run,
                                    feature.node_id: feature, attribute.node_id: attribute,
                                    preprocessing.node_id: preprocessing})
        return self

    def get_connected_nodes(self, node, depth=1):
        connected_nodes = {}
        search_list = node.connections
        while depth > 0:
            depth -= 1
            temporary_search_list = []
            for connection in search_list:
                if connection not in connected_nodes and connection is not node.node_id and connection in self._node_dict:
                    connected_nodes.update({self._node_dict[connection].node_id: self._node_dict[connection]})
                    for following_connection in self._node_dict[connection].connections:
                        if following_connection not in connected_nodes:
                            temporary_search_list.append(following_connection)
                    temporary_search_list.extend(self._node_dict[connection].connections)
            search_list = temporary_search_list
        return connected_nodes

    def add_nodes(self, nodes):
        self._node_dict.update(nodes)

    def add_node(self, node):
        self._node_dict.update({node.get_node_id(): node})

    def get_node(self, node_id):
        if node_id not in self._node_dict:
            return None
        return self._node_dict[node_id]

    def get_nodes(self, node_id_list):
        node_dict = {}
        for node_id in node_id_list:
            if self.check_if_node_exists(node_id):
                node_dict.update({node_id: self.get_node(node_id)})
        return node_dict

    def check_if_node_exists(self, node_id):
        return node_id in self._node_dict

    def get_ontology_structure(self):
        node_list = []
        node_structure = {}
        used_node_classes = []
        for node in self._node_dict.values():
            if node.node_class_id not in used_node_classes:
                used_node_classes.append(node.node_class_id)
                node_structure = {"Node": node.node_class_id, "Explanation": node.explanation,
                                  "Connections": ", ".join([f"{edge} {connection}" for connection, edge in
                                                            zip(node.get_class_connections()[0],
                                                                node.get_class_connections()[1])])}
                annotation_list = []
                annotations_to_ignore = {"connections", "class_connections", "node_id",
                                         "node_class_id", "explanation"}
                for key, value in node.__dict__.items():
                    if key not in annotations_to_ignore:
                        annotation_list.append(key)
                node_structure.update({"Annotations": annotation_list})
                node_list.append(node_structure)
        return str(node_list)

    def get_node_structure(self, node_list):
        node_structure_list = []
        for node in node_list:
            node_structure = {"Node": node.get_node_id(), "Explanation": node.get_explanation(),
                              "Connections": ", ".join([f"{edge} {connection}" for connection, edge in
                                                        zip(node.get_class_connections()[0],
                                                            node.get_class_connections()[1])])}
            annotation_list = []
            annotations_to_ignore = {"connections", "class_connections", "node_id",
                                     "node_class_id", "explanation"}
            for key, value in node.__dict__.items():
                if key not in annotations_to_ignore:
                    annotation_list.append(key)
            node_structure.update({"Annotations": annotation_list})
            node_structure_list.append(node_structure)
        return node_structure_list

    def get_ontology_node_overview(self):
        return list(self._node_dict.keys())

    def get_instances_by_class(self, node_class_id_list):
        instance_list = []
        for key, value in self._node_dict.items():
            if value.get_node_class_id() in node_class_id_list:
                instance_list.append(value)
        return instance_list

    def create_class_graph(self):
        g = nx.DiGraph()

        for node in self._node_dict.values():
            g.add_node(node.get_node_class_id())
            for connection in node.get_class_connections():
                g.add_edge(node.get_node_class_id(), connection)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=20)

        plt.title("Graphical Representation of Nodes and Connections")
        plt.show()

    def create_instance_graph(self):
        g = nx.DiGraph()

        for node_id, node in self._node_dict.items():
            g.add_node(node_id)
            for connection in node.get_node_connections():
                g.add_edge(node_id, connection)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(g)
        nx.draw(g, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=20)

        plt.title("Graphical Representation of Nodes and Connections")
        plt.show()

    def create_dynamic_class_graph(self):
        net = Network(height="100vh", width="100vw", directed=True)

        node_class_set = set()
        for node in self._node_dict.values():
            if node.get_node_class_id() not in node_class_set:
                net.add_node(node.get_node_class_id(), title=node.get_internal_structure())
                node_class_set.add(node.get_node_class_id())

        for node in self._node_dict.values():
            if node.get_node_class_id() in node_class_set:
                node_class_set.remove(node.get_node_class_id())
                for connection, edge in zip(node.get_class_connections()[0], node.get_class_connections()[1]):
                    net.add_edge(node.get_node_class_id(), connection, label=edge, arrows="to", length=200)

        output_file = "class_graph.html"
        net.save_graph(output_file)
        _insert_headline(headline="Ontology Node Class Diagram", output_file=output_file)

        webbrowser.open(output_file)

    def create_dynamic_instance_graph(self):
        net = Network(height="100vh", width="100vw", directed=True)

        used_nodes_list = []
        for node_id in self._node_dict.keys():
            used_nodes_list.append(node_id)
            for connection in self.get_node(node_id).connections:
                used_nodes_list.append(connection)
        unique_used_nodes_list = list(set(used_nodes_list))
        for unique_node_id in unique_used_nodes_list:
            net.add_node(unique_node_id)

        for node_id, node in self._node_dict.items():
            for connection in node.get_node_connections():
                net.add_edge(node_id, connection)

        output_file = "instance_graph.html"
        net.save_graph(output_file)
        _insert_headline(headline="Ontology Node Instance Diagram", output_file=output_file)

        webbrowser.open(output_file)


class Node:
    class_connections = [[], []]
    node_class_id = ""
    explanation = ""

    def __init__(self, node_id, connections):
        self.node_id = node_id
        self.connections = connections

    def get_node_id(self):
        return self.node_id

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    @classmethod
    def get_class_connections(cls):
        return cls.class_connections

    @classmethod
    def get_node_class_id(cls):
        return cls.node_class_id

    def get_node_class_id_from_instance(self):
        return self.node_class_id

    @classmethod
    def get_explanation(cls):
        return cls.explanation

    def get_node_connections(self):
        return self.connections

    def get_internal_structure(self):
        return list(self.__dict__.keys())

    def get_data(self):
        return self.__dict__

    def __str__(self):
        attributes = ', '.join(f'{key}={value}' for key, value in self.__dict__.items())
        return f'{self.__class__.__name__}({attributes})'

    @classmethod
    def generate_multiple_nodes(cls, n, seed=42):
        if seed is not None:
            random.seed(seed)  # Set the seed for reproducibility
        generated_models = [cls.generate_random_node() for _ in range(n)]
        generated_models.append(cls.get_premade_node())
        return generated_models

    @classmethod
    def generate_random_node(cls):
        pass

    @classmethod
    def get_premade_node(cls):
        pass


class GenericNode(Node):
    def __init__(self, node_id, node_class, connections, **kwargs):
        super().__init__(node_id, connections)
        self.node_class_id = node_class.get_node_class_id()
        self.class_connections = node_class.get_class_connections()
        self.explanation = node_class.get_explanation()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_node_class_id(self):
        return self.node_class_id

    def get_class_connections(self):
        return self.class_connections


class GenericClass:
    def __init__(self, node_class_id, class_connections, explanation):
        self.node_class_id = node_class_id
        self.class_connections = class_connections
        self.explanation = explanation

    def get_class_connections(self):
        return self.class_connections

    def get_definition(self):
        return self.explanation

    def get_node_class_id(self):
        return self.node_class_id

    def get_explanation(self):
        return self.explanation

    def update_class_connections(self, class_connections):
        self.class_connections = class_connections


class Model(Node):
    class_connections = [["Dataset", "TrainingRun"], ["trainedWith", "trainedBy"]]
    node_class_id = "Model"
    explanation = "A model is an algorithm trained on data."

    def __init__(self, node_id, connections, algorithm, accuracy, giniIndex, precision, recall, f1Score,
                 confusionMatrix, rocAucScore, trainedWith, trainedBy):
        super().__init__(node_id, connections)
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.giniIndex = giniIndex
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.confusionMatrix = confusionMatrix
        self.rocAucScore = rocAucScore
        self.trainedWith = trainedWith
        self.trainedBy = trainedBy
        self.explanation = Model.explanation
        self.class_connections = Model.class_connections
        self.node_class_id = Model.node_class_id

    @classmethod
    def get_premade_node(cls):
        return Model(node_id='model_1', algorithm='DecisionTree', accuracy=0.95768,
                     giniIndex=0.042319749216300995, precision={'Class 0': 0.95, 'Class 1': 0.96},
                     recall={'Class 0': 0.94, 'Class 1': 0.97}, f1Score={'Class 0': 0.95, 'Class 1': 0.96},
                     confusionMatrix=[[239, 14], [13, 372]], rocAucScore=0.9587,
                     trainedWith='niryo_dataset_september_2024', trainedBy='trainingRun_76d864c9',
                     connections=['niryo_dataset_september_2024', 'training_run_76d864c9'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"model_{random.randint(10000000, 99999999)}"
        node_class = 'Model'
        algorithm = random.choice(['randomForest', 'SVM', 'neuralNetwork', 'logisticRegression'])
        accuracy = round(random.uniform(0.5, 0.9), 2)
        giniIndex = round(random.uniform(0.1, 0.4), 2)
        precision = {'Class 0': round(random.uniform(0.6, 0.8), 2), 'Class 1': round(random.uniform(0.5, 0.7), 2)}
        recall = {'Class 0': round(random.uniform(0.6, 0.8), 2), 'Class 1': round(random.uniform(0.5, 0.7), 2)}
        f1Score = {'Class 0': round(random.uniform(0.6, 0.8), 2), 'Class 1': round(random.uniform(0.5, 0.7), 2)}
        confusionMatrix = [[random.randint(50, 100), random.randint(10, 50)],
                           [random.randint(30, 70), random.randint(30, 70)]]
        rocAucScore = round(random.uniform(0.6, 0.8), 2)
        trainedWith = f'dataset_{random.randint(10000000, 99999999)}'
        trainedBy = f'trainingRun_{random.randint(10000000, 99999999)}'
        connections = [trainedWith, trainedBy]

        return Model(node_id=node_id, node_class_id=node_class, algorithm=algorithm, accuracy=accuracy,
                     giniIndex=giniIndex, precision=precision, recall=recall, f1Score=f1Score,
                     confusionMatrix=confusionMatrix, rocAucScore=rocAucScore,
                     trainedWith=trainedWith, trainedBy=trainedBy, connections=connections)


class TrainingRun(Node):
    class_connections = [["Dataset", "Model"], ["hasInput", "hasOutput"]]
    node_class_id = "TrainingRun"

    def __init__(self, node_id, date, purpose, hasInput, hasOutput, connections):
        super().__init__(node_id, connections)
        self.date = date
        self.purpose = purpose
        self.hasInput = hasInput
        self.hasOutput = hasOutput
        self.explanation = "A training run is a group of jointly trained models that are all based on a specific data set."
        self.class_connections = TrainingRun.class_connections
        self.node_class_id = TrainingRun.node_class_id

    @classmethod
    def get_premade_node(cls):
        return TrainingRun(node_id='training_run_1', date='10.10.2024',
                           purpose='Train models to predict if a screw can be lifted to a specific position by a roboter arm',
                           hasInput='niryo_dataset_september_2024', hasOutput='model_1',
                           connections=['niryo_dataset_september_2024', 'model_1'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"training_run_{random.randint(10000000, 99999999)}"
        node_class = 'TrainingRun'

        # Generate a random date between two dates
        start_date = datetime.date(2023, 1, 1)
        end_date = datetime.date(2024, 12, 31)
        time_between_dates = end_date - start_date
        random_number_of_days = random.randrange(time_between_dates.days)
        random_date = start_date + datetime.timedelta(days=random_number_of_days)
        date = random_date.strftime("%d.%m.%Y")

        # Random purpose from a predefined list
        purpose = random.choice(
            ['Train models to predict if a cylinder slides down a slide', 'Evaluate model performance on test data',
             'Optimize hyperparameters for model training', 'Run cross-validation experiments',
             'Preprocess data for training'])

        # Random IDs for inputs and outputs
        hasInput = f"dataset_{random.randint(10000000, 99999999)}"
        hasOutput = f"model_{random.randint(10000000, 99999999)}"

        # Connections based on hasInput and hasOutput
        connections = [hasInput, hasOutput]

        return TrainingRun(node_id=node_id, node_class_id=node_class, date=date, purpose=purpose, hasInput=hasInput,
                           hasOutput=hasOutput, connections=connections)


class Dataset(Node):
    class_connections = [["TrainingRun", "Feature"], ["usedBy", "hasFeature"]]
    node_class_id = "Dataset"

    def __init__(self, node_id, connections, amountOfRows, amountOfAttributes, usedBy,
                 dataType, domain, locationOfDataRecording, dateOfRecording, createdBy):
        super().__init__(node_id, connections)
        self.amountOfRows = amountOfRows
        self.amountOfAttributes = amountOfAttributes
        self.usedBy = usedBy
        self.dataType = dataType
        self.domain = domain
        self.locationOfDataRecording = locationOfDataRecording
        self.dateOfRecording = dateOfRecording
        self.createdBy = createdBy
        self.explanation = "A Dataset consisting of multiple Rows."
        self.class_connections = Dataset.class_connections
        self.node_class_id = Dataset.node_class_id

    @classmethod
    def get_premade_node(cls):
        return Dataset(node_id='niryo_dataset_september_2024', amountOfRows=2126,
                       amountOfAttributes=12,
                       usedBy='training_run_1',
                       dataType='Dataset', domain='Niryo Robot', locationOfDataRecording='RWU',
                       dateOfRecording='Q4 2024', createdBy='',
                       connections=['training_run_76d864c9', "Test Durchgang", "Schrauben ID", "Schraubentyp"])

    @classmethod
    def generate_random_node(cls):
        node_id = f"dataset_{random.randint(10000000, 99999999)}"
        node_class = 'Dataset'
        amountOfRows = random.randint(100, 1000)
        amountOfAttributes = random.randint(1, 10)
        usedBy = f"training_run_{random.randint(10000000, 99999999)}"
        hasFeature = f"feature_{random.randint(10000000, 99999999)}"
        contains = f"data_row_{random.randint(10000000, 99999999)}"
        dataType = 'Dataset'
        domain = random.choice(['FestoSystem', 'OtherDomain'])
        locationOfDataRecording = random.choice(['RWU', 'OtherLocation'])
        dateOfRecording = random.choice(['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'])
        createdBy = f"preprocessing_{random.randint(10000000, 99999999)}"
        hasLabel = f"label_{random.randint(10000000, 99999999)}"
        connections = [usedBy, contains, hasFeature]

        return cls(node_id=node_id, node_class_id=node_class, amountOfRows=amountOfRows,
                   amountOfAttributes=amountOfAttributes, usedBy=usedBy, hasFeature=hasFeature, contains=contains,
                   dataType=dataType, domain=domain, locationOfDataRecording=locationOfDataRecording,
                   dateOfRecording=dateOfRecording, createdBy=createdBy, hasLabel=hasLabel, connections=connections)


# Todo Fill out
class DataRow(Node):
    pass


class Attribute(Node):
    class_connections = [["Preprocessing", "Dataset"], ["usedBy", "partOf"]]
    node_class_id = "Attribute"

    def __init__(self, node_id, connections, attributeName, usedBy, partOf):
        super().__init__(node_id, connections)
        self.attributeName = attributeName
        self.usedBy = usedBy
        self.partOf = partOf
        self.explanation = "An attribute is a characteristic of a dataset."
        self.class_connections = Attribute.class_connections

    @classmethod
    def get_premade_node(cls):
        return Attribute(node_id='attribute_a11be75b', attributeName='Result',
                         usedBy='Preprocessing_b9875fe0', partOf='Dataset_58ddb600',
                         connections=['preprocessing_b9875fe0', 'dataset_58ddb600'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"attribute_{random.randint(10000000, 99999999)}"
        node_class = 'Attribute'
        attribute_name = random.choice(['Result', 'Category', 'Value', 'Score', 'Label'])
        used_by = f"preprocessing_{random.randint(10000000, 99999999)}"
        part_of = f"dataset_{random.randint(10000000, 99999999)}"
        connections = [used_by, part_of]

        return cls(node_id=node_id, node_class_id=node_class, attributeName=attribute_name, usedBy=used_by,
                   partOf=part_of,
                   connections=connections)


class Preprocessing(Node):
    class_connections = [["Attribute"], ["hasInput"]]
    node_class_id = "Preprocessing"

    def __init__(self, node_id, connections, hasInput, pythonFile):
        super().__init__(node_id, connections)
        self.hasInput = hasInput
        self.pythonFile = pythonFile
        self.explanation = "Preprocessing is the process of preparing data for machine learning algorithms. It involves cleaning, transforming, and encoding data to make it suitable for training models."
        self.class_connections = Preprocessing.class_connections
        self.node_class_id = Preprocessing.node_class_id

    @classmethod
    def get_premade_node(cls):
        return Preprocessing(node_id='b9875fe0',
                             hasInput=['attribute_a11be75b', 'attribute_4987a624'], pythonFile="""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pickle
df = pd.read_csv("sparqlResult.csv")
df
df.drop('experiment', axis='columns', inplace=True)
df.drop('cylinder', axis='columns', inplace=True)
df
X = df.drop('result', axis=1)
y = df['result']
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, y)
cross_val_score(clf, X, y, cv=5)
plt.figure(figsize=(10,8))
plot_tree(clf, feature_names=X.columns, class_names=['True', 'False'], filled=True)
plt.savefig('FestoDecisionTree.png', dpi=800)
plt.show()
pickle.dump(clf, open('decisionTree.pickle', 'wb'))
y_pred = clf.predict(X)
y_pred
""", connections=[])

    @classmethod
    def generate_random_node(cls):
        node_id = f"preprocessing_{random.randint(10000000, 99999999)}"
        node_class = 'Preprocessing'

        # Generate random inputs (attributes)
        has_input = [f"attribute_{random.randint(10000000, 99999999)}" for _ in range(random.randint(1, 5))]

        # Python script remains the same as the provided one
        python_file = """
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pickle
df = pd.read_csv("sparqlResult.csv")
df
df.drop('experiment', axis='columns', inplace=True)
df.drop('cylinder', axis='columns', inplace=True)
df
X = df.drop('result', axis=1)
y = df['result']
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, y)
cross_val_score(clf, X, y, cv=5)
plt.figure(figsize=(10,8))
plot_tree(clf, feature_names=X.columns, class_names=['True', 'False'], filled=True)
plt.savefig('FestoDecisionTree.png', dpi=800)
plt.show()
pickle.dump(clf, open('decisionTree.pickle', 'wb'))
y_pred = clf.predict(X)
y_pred
"""
        # Connections can be derived from inputs or other logic
        connections = has_input[:]

        return cls(node_id=node_id, node_class_id=node_class, hasInput=has_input, pythonFile=python_file,
                   connections=connections)


class Feature(Node):
    class_connections = [["Dataset"], ["partOf"]]
    node_class_id = "Feature"

    def __init__(self, node_id, node_class_id, connections, feature_name, datatype, mean, standard_deviation, minimum,
                 maximum):
        super().__init__(node_id, connections)
        self.featureName = feature_name
        self.datatype = datatype
        self.mean = mean
        self.standardDeviation = standard_deviation
        self.minimum = minimum
        self.maximum = maximum
        self.explanation = "A feature is based on an attribute and is part of a dataset."
        self.class_connections = Feature.class_connections
        self.node_class_id = Feature.node_class_id

    @classmethod
    def get_premade_node(cls):
        return Feature(node_id='feature_87176016', node_class_id='Feature', feature_name='Pressure',
                       datatype='numerical',
                       mean=3.003, standard_deviation=0.310, minimum=2.45, maximum=4.91,
                       connections=['data_row_fc9f70a3'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"feature_{random.randint(10000000, 99999999)}"
        node_class = 'Feature'
        feature_name = random.choice(['Pressure', 'Temperature', 'Volume', 'Mass', 'Velocity'])
        datatype = random.choice(['numerical', 'categorical'])

        # Generate random values for mean, standard deviation, minimum, and maximum
        mean = round(random.uniform(1.0, 5.0), 3)
        standard_deviation = round(random.uniform(0.1, 1.0), 3)
        minimum = round(random.uniform(0.0, mean), 3)
        maximum = round(random.uniform(mean, 6.0), 3)

        connections = [f"data_row_{random.randint(10000000, 99999999)}"]

        return cls(node_id=node_id, node_class_id=node_class, feature_name=feature_name, datatype=datatype, mean=mean,
                   standard_deviation=standard_deviation, minimum=minimum, maximum=maximum, connections=connections)
