import datetime
import random

import matplotlib.pyplot as plt
import networkx as nx


class Ontology:

    def __init__(self):
        self._node_dict = {}

    def create_demo_ontology(self, model_count=3):
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

    def get_node(self, node_id):
        if node_id.lower() not in self._node_dict:
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
        for node in self._node_dict.values():
            if node not in node_list:
                node_structure = {"Node": node.node_class, "Explanation": node.explanation,
                                  "Connections": [cls.__name__ for cls in node.get_class_connections()]}
            annotation_list = []
            for key, value in node.__dict__.items():
                annotation_list.append(key)
            node_structure.update({"Annotations": annotation_list})
            node_list.append(node_structure)
        return str(node_list)

    def get_ontology_node_overview(self):
        return list(self._node_dict.keys())

    def create_class_graph(self):
        # Define the nodes and their connections
        nodes = [{'Node': 'DataSet', 'Connections': ['TrainingRun', 'Feature']},
                 {'Node': 'Model', 'Connections': ['DataSet', 'TrainingRun']},
                 {'Node': 'TrainingRun', 'Connections': ['DataSet', 'Model']}, {'Node': 'Feature', 'Connections': []},
                 {'Node': 'Attribute', 'Connections': ['DataSet']}, {'Node': 'Preprocessing', 'Connections': []}, ]

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes and edges to the graph
        for node in nodes:
            G.add_node(node['Node'])
            for connection in node['Connections']:
                G.add_edge(node['Node'], connection)

        # Draw the graph
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold',
                edge_color='gray', arrowsize=20)

        # Display the graph
        plt.title("Graphical Representation of Nodes and Connections")
        plt.show()

    def create_instance_graph(self):
        pass


class Node:
    def __init__(self, node_id, node_class, connections):
        self.class_connections = None
        self.node_id = node_id
        self.node_class = node_class
        self.connections = connections

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def get_class(self):
        return self.node_class

    def get_data(self):
        return {'node_id': self.node_id, 'node_class': self.node_class, 'connections': self.connections, }

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

    def get_class_connections(self):
        return self.class_connections


class Model(Node):
    def __init__(self, node_id, node_class, connections, algorithm, accuracy, giniIndex, precision, recall, f1Score,
                 confusionMatrix, truePositivesClass1, trueNegativesClass0, falsePositivesClass0, falseNegativesClass1,
                 rocAucScore, crossValidationScores, mean, standardDeviation, trainedWith, trainedBy):
        super().__init__(node_id, node_class, connections)
        self.algorithm = algorithm
        self.accuracy = accuracy
        self.giniIndex = giniIndex
        self.precision = precision
        self.recall = recall
        self.f1Score = f1Score
        self.confusionMatrix = confusionMatrix
        self.truePositivesClass1 = truePositivesClass1
        self.trueNegativesClass0 = trueNegativesClass0
        self.falsePositivesClass0 = falsePositivesClass0
        self.falseNegativesClass1 = falseNegativesClass1
        self.rocAucScore = rocAucScore
        self.crossValidationScores = crossValidationScores
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.trainedWith = trainedWith
        self.trainedBy = trainedBy
        self.explanation = "A model is an algorithm trained on data."
        self.class_connections = [Dataset, TrainingRun]

    def get_data(self):
        return {'id': self.node_id, 'algorithm': self.algorithm, 'accuracy': self.accuracy, 'giniIndex': self.giniIndex,
                'precision': self.precision, 'recall': self.recall, 'f1Score': self.f1Score,
                'confusionMatrix': self.confusionMatrix, 'truePositivesClass1': self.truePositivesClass1,
                'trueNegativesClass0': self.trueNegativesClass0, 'falsePositivesClass0': self.falsePositivesClass0,
                'falseNegativesClass1': self.falseNegativesClass1, 'rocAucScore': self.rocAucScore,
                'crossValidationScores': self.crossValidationScores, 'mean': self.mean,
                'standardDeviation': self.standardDeviation, 'trainedWith': self.trainedWith,
                'trainedBy': self.trainedBy,
                'connections': self.connections}

    @classmethod
    def get_premade_node(cls):
        return Model(node_id='model_a2f6fb37', node_class='Model', algorithm='randomForest', accuracy=0.6,
                     giniIndex=0.2, precision={'Class 0': 0.65, 'Class 1': 0.55},
                     recall={'Class 0': 0.70, 'Class 1': 0.50}, f1Score={'Class 0': 0.67, 'Class 1': 0.52},
                     confusionMatrix=[[70, 30], [50, 50]], truePositivesClass1=50, trueNegativesClass0=70,
                     falsePositivesClass0=30, falseNegativesClass1=50, rocAucScore=0.65,
                     crossValidationScores={'fold 1': 0.58, 'fold 2': 0.62, 'fold 3': 0.59, 'fold 4': 0.61,
                                            'fold 5': 0.60}, mean=0.60, standardDeviation=0.015,
                     trainedWith='Dataset_58ddb600', trainedBy='TrainingRun_76d864c9',
                     connections=['dataset_58ddb600', 'training_run_76d864c9'])

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
        truePositivesClass1 = confusionMatrix[1][1]
        trueNegativesClass0 = confusionMatrix[0][0]
        falsePositivesClass0 = confusionMatrix[0][1]
        falseNegativesClass1 = confusionMatrix[1][0]
        rocAucScore = round(random.uniform(0.6, 0.8), 2)
        crossValidationScores = {f'fold {i}': round(random.uniform(0.55, 0.65), 2) for i in range(1, 6)}
        mean = round(random.uniform(0.55, 0.65), 2)
        standardDeviation = round(random.uniform(0.01, 0.03), 3)
        trainedWith = f'Dataset_{random.randint(10000000, 99999999)}'
        trainedBy = f'TrainingRun_{random.randint(10000000, 99999999)}'
        connections = [trainedWith, trainedBy]

        return Model(node_id=node_id, node_class=node_class, algorithm=algorithm, accuracy=accuracy,
                     giniIndex=giniIndex, precision=precision, recall=recall, f1Score=f1Score,
                     confusionMatrix=confusionMatrix, truePositivesClass1=truePositivesClass1,
                     trueNegativesClass0=trueNegativesClass0, falsePositivesClass0=falsePositivesClass0,
                     falseNegativesClass1=falseNegativesClass1, rocAucScore=rocAucScore,
                     crossValidationScores=crossValidationScores, mean=mean, standardDeviation=standardDeviation,
                     trainedWith=trainedWith, trainedBy=trainedBy, connections=connections)


class TrainingRun(Node):
    def __init__(self, node_id, node_class, date, purpose, hasInput, hasOutput, connections):
        super().__init__(node_id, node_class, connections)
        self.date = date
        self.purpose = purpose
        self.hasInput = hasInput
        self.hasOutput = hasOutput
        self.explanation = "A training run is a group of jointly trained models that are all based on a specific data set."
        self.class_connections = [Dataset, Model]

    def get_data(self):
        return {'id': self.id, 'date': self.date, 'purpose': self.purpose, 'hasInput': self.hasInput,
                'hasOutput': self.hasOutput, 'connections': self.connections}

    @classmethod
    def get_premade_node(cls):
        return TrainingRun(node_id='training_run_76d864c9', node_class='TrainingRun', date='01.01.2024',
                           purpose='Train models to predict if a cylinder slides down a slide',
                           hasInput='DataSet_58ddb600', hasOutput='Model_a2f6fb37',
                           connections=['dataset_58ddb600', 'model_a2f6fb37'])

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
        hasInput = f"DataSet_{random.randint(10000000, 99999999)}"
        hasOutput = f"Model_{random.randint(10000000, 99999999)}"

        # Connections based on hasInput and hasOutput
        connections = [hasInput, hasOutput]

        return TrainingRun(node_id=node_id, node_class=node_class, date=date, purpose=purpose, hasInput=hasInput,
                           hasOutput=hasOutput, connections=connections)


class Dataset(Node):

    def __init__(self, node_id, node_class, connections, amountOfRows, amountOfAttributes, usedBy, hasFeature, contains,
                 dataType, domain, locationOfDataRecording, dateOfRecording, createdBy, hasLabel):
        super().__init__(node_id, node_class, connections)
        self.amountOfRows = amountOfRows
        self.amountOfAttributes = amountOfAttributes
        self.usedBy = usedBy
        self.hasFeature = hasFeature
        self.contains = contains
        self.dataType = dataType
        self.domain = domain
        self.locationOfDataRecording = locationOfDataRecording
        self.dateOfRecording = dateOfRecording
        self.createdBy = createdBy
        self.hasLabel = hasLabel
        self.explanation = "A Dataset consisting of multiple Rows."
        self.class_connections = [TrainingRun, DataRow, Feature]

    def get_data(self):
        return {'id': self.id, 'amountOfRows': self.amountOfRows, 'amountOfAttributes': self.amountOfAttributes,
                'usedBy': self.usedBy, 'hasFeature': self.hasFeature, 'contains': self.contains,
                'dataType': self.dataType,
                'domain': self.domain, 'locationOfDataRecording': self.locationOfDataRecording,
                'dateOfRecording': self.dateOfRecording, 'createdBy': self.createdBy, 'hasLabel': self.hasLabel,
                'connections': self.connections}

    @classmethod
    def get_premade_node(cls):
        return Dataset(node_id='dataset_58ddb600', node_class='DataSet', amountOfRows=300, amountOfAttributes=4,
                       usedBy='training_run_76d864c9', hasFeature='Feature_87176016', contains='DataRow_fc9f70a3',
                       dataType='DataSet', domain='FestoSystem', locationOfDataRecording='RWU',
                       dateOfRecording='Q4 2023', createdBy='Preprocessing_b9875fe0', hasLabel='Label_fa5649f9',
                       connections=['training_run_76d864c9', 'data_row_fc9f70a3', 'feature_87176016'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"dataset_{random.randint(10000000, 99999999)}"
        node_class = 'DataSet'
        amountOfRows = random.randint(100, 1000)
        amountOfAttributes = random.randint(1, 10)
        usedBy = f"training_run_{random.randint(10000000, 99999999)}"
        hasFeature = f"feature_{random.randint(10000000, 99999999)}"
        contains = f"data_row_{random.randint(10000000, 99999999)}"
        dataType = 'DataSet'
        domain = random.choice(['FestoSystem', 'OtherDomain'])
        locationOfDataRecording = random.choice(['RWU', 'OtherLocation'])
        dateOfRecording = random.choice(['Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023'])
        createdBy = f"preprocessing_{random.randint(10000000, 99999999)}"
        hasLabel = f"label_{random.randint(10000000, 99999999)}"
        connections = [usedBy, contains, hasFeature]

        return cls(node_id=node_id, node_class=node_class, amountOfRows=amountOfRows,
                   amountOfAttributes=amountOfAttributes, usedBy=usedBy, hasFeature=hasFeature, contains=contains,
                   dataType=dataType, domain=domain, locationOfDataRecording=locationOfDataRecording,
                   dateOfRecording=dateOfRecording, createdBy=createdBy, hasLabel=hasLabel, connections=connections)


# Todo Fill out
class DataRow(Node):
    pass


class Attribute(Node):
    def __init__(self, node_id, node_class, connections, attributeName, usedBy, partOf):
        super().__init__(node_id, node_class, connections)
        self.attributeName = attributeName
        self.usedBy = usedBy
        self.partOf = partOf
        self.explanation = "An attribute is a characteristic of a dataset."
        self.class_connections = [Preprocessing, Dataset]

    def get_data(self):
        return {'attributeName': self.attributeName}

    @classmethod
    def get_premade_node(cls):
        return Attribute(node_id='attribute_a11be75b', node_class='Attribute', attributeName='Result',
                         usedBy='Preprocessing_b9875fe0', partOf='DataSet_58ddb600',
                         connections=['preprocessing_b9875fe0', 'dataset_58ddb600'])

    @classmethod
    def generate_random_node(cls):
        node_id = f"attribute_{random.randint(10000000, 99999999)}"
        node_class = 'Attribute'
        attribute_name = random.choice(['Result', 'Category', 'Value', 'Score', 'Label'])
        used_by = f"preprocessing_{random.randint(10000000, 99999999)}"
        part_of = f"dataset_{random.randint(10000000, 99999999)}"
        connections = [used_by, part_of]

        return cls(node_id=node_id, node_class=node_class, attributeName=attribute_name, usedBy=used_by, partOf=part_of,
                   connections=connections)


class Preprocessing(Node):
    def __init__(self, node_id, node_class, connections, hasInput, pythonFile):
        super().__init__(node_id, node_class, connections)
        self.hasInput = hasInput
        self.pythonFile = pythonFile
        self.explanation = "Preprocessing is the process of preparing data for machine learning algorithms. It involves cleaning, transforming, and encoding data to make it suitable for training models."
        self.class_connections = [Attribute]

    def get_data(self):
        return {'id': self.id, 'hasInput': self.hasInput, 'pythonFile': self.pythonFile}

    @classmethod
    def get_premade_node(cls):
        return Preprocessing(node_id='b9875fe0', node_class='Preprocessing',
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

        return cls(node_id=node_id, node_class=node_class, hasInput=has_input, pythonFile=python_file,
                   connections=connections)


class Feature(Node):
    def __init__(self, node_id, node_class, connections, feature_name, datatype, mean, standard_deviation, minimum,
                 maximum):
        super().__init__(node_id, node_class, connections)
        self.featureName = feature_name
        self.datatype = datatype
        self.mean = mean
        self.standardDeviation = standard_deviation
        self.minimum = minimum
        self.maximum = maximum
        self.explanation = "A feature is based on an attribute and is part of a dataset."
        self.class_connections = [DataRow]

    def get_data(self):
        return {'id': self.node_id, 'featureName': self.featureName, 'datatype': self.datatype, 'mean': self.mean,
                'standardDeviation': self.standardDeviation, 'minimum': self.minimum, 'maximum': self.maximum,
                'connections': self.connections}

    @classmethod
    def get_premade_node(cls):
        return Feature(node_id='feature_87176016', node_class='Feature', feature_name='Pressure', datatype='numerical',
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

        return cls(node_id=node_id, node_class=node_class, feature_name=feature_name, datatype=datatype, mean=mean,
                   standard_deviation=standard_deviation, minimum=minimum, maximum=maximum, connections=connections)
