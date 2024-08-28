class Ontology:

    def __init__(self):
        self.node_dict = {}

    def create_basic_ontology(self):
        data1 = Dataset.get_premade_object()
        training_run1 = TrainingRun.get_premade_object()
        model1 = Model.get_premade_object()
        feature1 = Feature.get_premade_object()
        attribute1 = Attribute.get_premade_object()
        preprocessing1 = Preprocessing.get_premade_object()

        self.node_dict.update({data1.node_id: data1, model1.node_id: model1, training_run1.node_id: training_run1,
                               feature1.node_id: feature1,
                               attribute1.node_id: attribute1, preprocessing1.node_id: preprocessing1})

        # self.label = Label.get_premade_object()
        return self

    def get_connected_nodes(self, node, depth=1):
        connected_nodes = {}
        search_list = node.connections
        while depth > 0:
            depth -= 1
            temporary_search_list = []
            for connection in search_list:
                if connection not in connected_nodes and connection is not node.node_id and connection in self.node_dict:
                    connected_nodes.update({self.node_dict[connection].node_id: self.node_dict[connection]})
                    for following_connection in self.node_dict[connection].connections:
                        if following_connection not in connected_nodes:
                            temporary_search_list.append(following_connection)
                    temporary_search_list.extend(self.node_dict[connection].connections)
            search_list = temporary_search_list
        return connected_nodes

    def add_nodes(self, nodes):
        self.node_dict.update(nodes)

    def get_node(self, node_id):
        if node_id not in self.node_dict:
            return None
        return self.node_dict[node_id]

    def get_ontology_structure(self):
        node_list = []
        for node in self.node_dict.values():
            connection_classes_list = []
            for connection in node.connections:
                found_connection = self.get_node(connection)
                if found_connection is not None:
                    connection_classes_list.append(found_connection.get_class())
            node_dict = {"Node": node.node_class, "Explanation": node.explanation,
                         "Connections": connection_classes_list}

            annotation_list = []
            for key, value in node.__dict__.items():
                annotation_list.append(key)
            node_dict.update({"Annotations": annotation_list})
            node_list.append(node_dict)
        return str(node_list)


class Node:
    def __init__(self, node_id, node_class, connections, explanation=""):
        self.node_id = node_id
        self.node_class = node_class
        self.connections = connections

    def get_class(self):
        return self.node_class

    def get_data(self):
        return {
            'node_id': self.node_id,
            'node_class': self.node_class,
            'connections': self.connections,
        }


class Model(Node):
    def __init__(self, node_id, node_class, connections, algorithm, accuracy, giniIndex, precision, recall, f1Score,
                 confusionMatrix,
                 truePositivesClass1, trueNegativesClass0, falsePositivesClass0, falseNegativesClass1, rocAucScore,
                 crossValidationScores, mean, standardDeviation, trainedWith, trainedBy):
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

    def get_data(self):
        return {
            'id': self.id,
            'algorithm': self.algorithm,
            'accuracy': self.accuracy,
            'giniIndex': self.giniIndex,
            'precision': self.precision,
            'recall': self.recall,
            'f1Score': self.f1Score,
            'confusionMatrix': self.confusionMatrix,
            'truePositivesClass1': self.truePositivesClass1,
            'trueNegativesClass0': self.trueNegativesClass0,
            'falsePositivesClass0': self.falsePositivesClass0,
            'falseNegativesClass1': self.falseNegativesClass1,
            'rocAucScore': self.rocAucScore,
            'crossValidationScores': self.crossValidationScores,
            'mean': self.mean,
            'standardDeviation': self.standardDeviation,
            'trainedWith': self.trainedWith,
            'trainedBy': self.trainedBy,
            'connections': self.connections
        }

    @classmethod
    def get_premade_object(cls):
        return Model(node_id='model_a2f6fb37',
                     node_class='Model',
                     algorithm='randomForest',
                     accuracy=0.6,
                     giniIndex=0.2,
                     precision={'Class 0': 0.65, 'Class 1': 0.55},
                     recall={'Class 0': 0.70, 'Class 1': 0.50},
                     f1Score={'Class 0': 0.67, 'Class 1': 0.52},
                     confusionMatrix=[[70, 30], [50, 50]],
                     truePositivesClass1=50,
                     trueNegativesClass0=70,
                     falsePositivesClass0=30,
                     falseNegativesClass1=50,
                     rocAucScore=0.65,
                     crossValidationScores={'fold 1': 0.58, 'fold 2': 0.62, 'fold 3': 0.59, 'fold 4': 0.61,
                                            'fold 5': 0.60},
                     mean=0.60,
                     standardDeviation=0.015,
                     trainedWith='Dataset_58ddb600',
                     trainedBy='TrainingRun_76d864c9',
                     connections=['dataset_58ddb600', 'training_run_76d864c9']
                     )


class TrainingRun(Node):
    def __init__(self, node_id, node_class, date, purpose, hasInput, hasOutput, connections):
        super().__init__(node_id, node_class, connections)
        self.date = date
        self.purpose = purpose
        self.hasInput = hasInput
        self.hasOutput = hasOutput
        self.explanation = "A training run is a group of jointly trained models that are all based on a specific data set."

    def get_data(self):
        return {
            'id': self.id,
            'date': self.date,
            'purpose': self.purpose,
            'hasInput': self.hasInput,
            'hasOutput': self.hasOutput,
            'connections': self.connections
        }

    @classmethod
    def get_premade_object(cls):
        return TrainingRun(node_id='training_run_76d864c9',
                           node_class='TrainingRun',
                           date='01.01.2024',
                           purpose='Train models to predict if a cylinder slides down a slide',
                           hasInput='DataSet_58ddb600',
                           hasOutput='Model_a2f6fb37',
                           connections=['dataset_58ddb600', 'model_a2f6fb37']
                           )


class Dataset(Node):

    def __init__(self, node_id, node_class, connections, amountOfRows, amountOfAttributes, usedBy, hasFeature, contains,
                 dataType, domain,
                 locationOfDataRecording, dateOfRecording, createdBy, hasLabel):
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

    def get_data(self):
        return {
            'id': self.id,
            'amountOfRows': self.amountOfRows,
            'amountOfAttributes': self.amountOfAttributes,
            'usedBy': self.usedBy,
            'hasFeature': self.hasFeature,
            'contains': self.contains,
            'dataType': self.dataType,
            'domain': self.domain,
            'locationOfDataRecording': self.locationOfDataRecording,
            'dateOfRecording': self.dateOfRecording,
            'createdBy': self.createdBy,
            'hasLabel': self.hasLabel,
            'connections': self.connections
        }

    @classmethod
    def get_premade_object(cls):
        return Dataset(node_id='dataset_58ddb600',
                       node_class='DataSet',
                       amountOfRows=300,
                       amountOfAttributes=4,
                       usedBy='training_run_76d864c9',
                       hasFeature='Feature_87176016',
                       contains='DataRow_fc9f70a3',
                       dataType='DataSet',
                       domain='FestoSystem',
                       locationOfDataRecording='RWU',
                       dateOfRecording='Q4 2023',
                       createdBy='Preprocessing_b9875fe0',
                       hasLabel='Label_fa5649f9',
                       connections=['training_run_76d864c9', 'data_row_fc9f70a3', 'feature_87176016']
                       )


class Attribute(Node):
    def __init__(self, node_id, node_class, connections, attributeName, usedBy, partOf):
        super().__init__(node_id, node_class, connections)
        self.attributeName = attributeName
        self.usedBy = usedBy
        self.partOf = partOf
        self.explanation = "An attribute is a characteristic of a dataset."

    def get_data(self):
        return {
            'attributeName': self.attributeName
        }

    @classmethod
    def get_premade_object(cls):
        return Attribute(node_id='attribute_a11be75b',
                         node_class='Attribute',
                         attributeName='Result',
                         usedBy='Preprocessing_b9875fe0',
                         partOf='DataSet_58ddb600',
                         connections=['preprocessing_b9875fe0', 'dataset_58ddb600']
                         )


class Preprocessing(Node):
    def __init__(self, node_id, node_class, connections, hasInput, pythonFile):
        super().__init__(node_id, node_class, connections)
        self.hasInput = hasInput
        self.pythonFile = pythonFile
        self.explanation = "Preprocessing is the process of preparing data for machine learning algorithms. It involves cleaning, transforming, and encoding data to make it suitable for training models."

    def get_data(self):
        return {
            'id': self.id,
            'hasInput': self.hasInput,
            'pythonFile': self.pythonFile
        }

    @classmethod
    def get_premade_object(cls):
        return Preprocessing(node_id='b9875fe0',
                             node_class='Preprocessing',
                             hasInput=['attribute_a11be75b', 'attribute_4987a624'],
                             pythonFile="""
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
""",
                             connections=[])


class Feature(Node):
    def __init__(self, node_id, node_class, connections, featureName, datatype, mean, standardDeviation, minimum,
                 maximum):
        super().__init__(node_id, node_class, connections)
        self.featureName = featureName
        self.datatype = datatype
        self.mean = mean
        self.standardDeviation = standardDeviation
        self.minimum = minimum
        self.maximum = maximum
        self.explanation = "A feature is based on an attribute and is part of a dataset."

    def get_data(self):
        return {
            'id': self.node_id,
            'featureName': self.featureName,
            'datatype': self.datatype,
            'mean': self.mean,
            'standardDeviation': self.standardDeviation,
            'minimum': self.minimum,
            'maximum': self.maximum,
            'connections': self.connections
        }

    @classmethod
    def get_premade_object(cls):
        return Feature(node_id='feature_87176016',
                       node_class='Feature',
                       featureName='Pressure',
                       datatype='numerical',
                       mean=3.003,
                       standardDeviation=0.310,
                       minimum=2.45,
                       maximum=4.91,
                       connections=['data_row_fc9f70a3']
                       )
