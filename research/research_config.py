from research.owl import *
from research.questionnaire import Questionnaire


class Initialization():
    def __init__(self, demo_mode=True):
        self.owl = Ontology()
        self.owl.create_demo_ontology()
        self.questionnaire = Questionnaire(demo_mode)

        # Create Class Ontology
        screw_class_node = GenericClass("Screw", class_connections=[], explanation="A metal Screw")
        test_case_class_node = GenericClass("TestCase", class_connections=[],
                                            explanation="A screw is given to a roboter arm. The roboter arm has to lift the screw and place it in a given place on a given angle.")
        robot_arm_class_node = GenericClass("Robotarm", class_connections=["Gripper"],
                                            explanation="A robot arm with different modules.")
        gripper_class_node = GenericClass("Gripper", class_connections=["Robotarm"], explanation="")
        feature_class_node = GenericClass("Feature", class_connections=["Dataset"],
                                          explanation="Feature of a Dataset with statistics.")
        self.owl.add_node(GenericNode("Niryo", robot_arm_class_node, connections=["training_run_1"]))
        self.owl.add_node(GenericNode('Silicone_gripper', gripper_class_node, connections=["Niryo"], typ="silicone"))
        self.owl.add_node(
            GenericNode('screw_530', screw_class_node, ["training_run_1"], length=30,
                        width=8,
                        thickness=3.5,
                        weight=5.1,
                        diameter=5,
                        coating=False))
        self.owl.add_node(GenericNode(node_id="Test Durchgang", node_class=feature_class_node, connections=["niryo_dataset_september_2024"], datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schrauben ID", node_class=feature_class_node,
                                      connections=["niryo_dataset_september_2024"], datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schraubentyp", node_class=feature_class_node,
                                      connections=["niryo_dataset_september_2024"], datatype="Enum/ Kategorisch"))
        self.owl.add_node(GenericNode(node_id="TestCase1", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicone_gripper"], angle=0, success=True))

    # self.owl.add_node(GenericNode(node_id=""))

    def get_questionnaire(self):
        return self.questionnaire

    def get_ontology(self):
        return self.owl
