import unittest
from unittest.mock import patch

from research.functions import rag
from research.owl import Ontology, Model, TrainingRun, Dataset, Attribute, Preprocessing, Feature


class TestOntology(unittest.TestCase):
    demo_ontology = None
    def setUp(self):
        # This runs before every test

        demo_ontology = Ontology()
        demo_ontology.create_demo_ontology()

    def test_read_data_entries(self):
        rag(self.demo_ontology, "Which Dataset did model_a2f6fb37 used and how many entries does it have?")
        self.assertEqual("The model with node ID `model_a2f6fb37` used the dataset with node ID `dataset_58ddb600`. This dataset contains 300 entries (rows).")

    def test_create_demo_ontology(self):
        # Test creating a demo ontology with default model count
        self.ontology.create_demo_ontology()
        self.assertEqual(len(self.ontology._node_dict),
                         24)  # 3 * 6 (Model, Dataset, TrainingRun, Feature, Attribute, Preprocessing) + 6 predefined

    def test_get_connected_nodes(self):
        # Test retrieving connected nodes with depth
        owl = self.ontology.create_demo_ontology()
        model = next(node for node in owl.ontology._node_dict.values() if isinstance(node, Model))
        connected_nodes = owl.ontology.get_connected_nodes(model, depth=1)
        self.assertGreater(len(connected_nodes), 0)

    def test_add_nodes(self):
        # Test adding nodes to the ontology
        node = Model.get_premade_node()
        self.ontology.add_nodes({node.node_id: node})
        self.assertIn(node.node_id, self.ontology._node_dict)

    def test_get_node(self):
        # Test retrieving a node by ID
        node = Model.get_premade_node()
        self.ontology.add_nodes({node.node_id: node})
        retrieved_node = self.ontology.get_node(node.node_id.lower())
        self.assertEqual(retrieved_node.node_id, node.node_id)

    def test_get_ontology_structure(self):
        # Test the structure of the ontology
        self.ontology.create_demo_ontology()
        structure = self.ontology.get_ontology_structure()
        self.assertIn("Node", structure)

    def test_get_ontology_node_overview(self):
        # Test the overview of ontology nodes
        self.ontology.create_demo_ontology()
        overview = self.ontology.get_ontology_node_overview()
        self.assertEqual(len(overview), 24)

    @patch('matplotlib.pyplot.show')
    def test_create_class_graph(self, mock_show):
        # Test creating a class graph (mocking plt.show to avoid displaying the graph)
        self.ontology.create_class_graph()
        mock_show.assert_called_once()


class TestModel(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade Model node
        node = Model.get_premade_node()
        self.assertEqual(node.node_id, 'model_a2f6fb37')
        self.assertEqual(node.algorithm, 'randomForest')

    def test_generate_random_node(self):
        # Test generating a random Model node
        node = Model.generate_random_node()
        self.assertIsInstance(node, Model)
        self.assertGreaterEqual(node.accuracy, 0.5)
        self.assertLessEqual(node.accuracy, 0.9)


class TestTrainingRun(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade TrainingRun node
        node = TrainingRun.get_premade_node()
        self.assertEqual(node.node_id, 'training_run_76d864c9')
        self.assertEqual(node.purpose, 'Train models to predict if a cylinder slides down a slide')

    def test_generate_random_node(self):
        # Test generating a random TrainingRun node
        node = TrainingRun.generate_random_node()
        self.assertIsInstance(node, TrainingRun)
        self.assertIn(node.purpose, [
            'Train models to predict if a cylinder slides down a slide',
            'Evaluate model performance on test data',
            'Optimize hyperparameters for model training',
            'Run cross-validation experiments',
            'Preprocess data for training'
        ])


class TestDataset(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade Dataset node
        node = Dataset.get_premade_node()
        self.assertEqual(node.node_id, 'dataset_58ddb600')
        self.assertEqual(node.amountOfRows, 300)

    def test_generate_random_node(self):
        # Test generating a random Dataset node
        node = Dataset.generate_random_node()
        self.assertIsInstance(node, Dataset)
        self.assertGreaterEqual(node.amountOfRows, 100)
        self.assertLessEqual(node.amountOfRows, 1000)


class TestAttribute(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade Attribute node
        node = Attribute.get_premade_node()
        self.assertEqual(node.node_id, 'attribute_a11be75b')
        self.assertEqual(node.attributeName, 'Result')

    def test_generate_random_node(self):
        # Test generating a random Attribute node
        node = Attribute.generate_random_node()
        self.assertIsInstance(node, Attribute)
        self.assertIn(node.attributeName, ['Result', 'Category', 'Value', 'Score', 'Label'])


class TestPreprocessing(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade Preprocessing node
        node = Preprocessing.get_premade_node()
        self.assertEqual(node.node_id, 'b9875fe0')
        self.assertEqual(node.hasInput, ['attribute_a11be75b', 'attribute_4987a624'])

    def test_generate_random_node(self):
        # Test generating a random Preprocessing node
        node = Preprocessing.generate_random_node()
        self.assertIsInstance(node, Preprocessing)
        self.assertGreaterEqual(len(node.hasInput), 1)
        self.assertLessEqual(len(node.hasInput), 5)


class TestFeature(unittest.TestCase):

    def test_get_premade_node(self):
        # Test the premade Feature node
        node = Feature.get_premade_node()
        self.assertEqual(node.node_id, 'feature_87176016')
        self.assertEqual(node.featureName, 'Pressure')

    def test_generate_random_node(self):
        # Test generating a random Feature node
        node = Feature.generate_random_node()
        self.assertIsInstance(node, Feature)
        self.assertIn(node.datatype, ['numerical', 'categorical'])


if __name__ == '__main__':
    unittest.main()
