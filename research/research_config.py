from research.owl import *
from research.questionnaire import Questionnaire


class Initialization():
    def __init__(self, demo_mode=True):
        self.owl = Ontology()
        self.owl.create_demo_ontology()
        self.questionnaire = Questionnaire(demo_mode)

        # Create Class Ontology
        screw_class_node = GenericClass("Screw", class_connections=[[], []], explanation="A metal Screw")
        test_case_class_node = GenericClass("TestCase", class_connections=[[], []],
                                            explanation="A screw is given to a roboter arm. The roboter arm has to "
                                                        "lift the screw and place it in a given place on a given angle.")
        robot_arm_class_node = GenericClass("Robotarm", class_connections=[["Gripper"], ["has"]],
                                            explanation="A robot arm with different modules.")
        gripper_class_node = GenericClass("Gripper", class_connections=[["Robotarm"], ["partOf"]], explanation="")
        feature_class_node = GenericClass("Feature", class_connections=[["Dataset"], ["belongsTo"]],
                                          explanation="Feature of a Dataset with statistics.")

        self.owl.add_node(GenericNode("Niryo", robot_arm_class_node, connections=["training_run_1"]))
        self.owl.add_node(GenericNode('Silicone_gripper', gripper_class_node, connections=["Niryo"], typ="silicone"))
        self.owl.add_node(GenericNode('Notch_gripper', gripper_class_node, connections=["Niryo"], typ="silicone"))
        self.owl.add_node(
            GenericNode('screw_530', screw_class_node, ["training_run_1"], length=30,
                        width=8,
                        thickness=3.5,
                        weight=5.1,
                        diameter=5.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_540', screw_class_node, ["training_run_1"], length=40,
                        width=8,
                        thickness=3.5,
                        weight=6.4,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_550', screw_class_node, ["training_run_1"], length=50,
                        width=8,
                        thickness=3.5,
                        weight=7.6,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_560', screw_class_node, ["training_run_1"], length=60,
                        width=8,
                        thickness=3.5,
                        weight=8.8,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_570', screw_class_node, ["training_run_1"], length=70,
                        width=8,
                        thickness=3.5,
                        weight=10.2,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_630', screw_class_node, ["training_run_1"], length=30,
                        width=10,
                        thickness=4.0,
                        weight=7.5,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_640', screw_class_node, ["training_run_1"], length=40,
                        width=10,
                        thickness=4.0,
                        weight=9.2,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_650', screw_class_node, ["training_run_1"], length=50,
                        width=10,
                        thickness=4.0,
                        weight=11.0,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_660', screw_class_node, ["training_run_1"], length=60,
                        width=10,
                        thickness=4.0,
                        weight=12.7,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, ["training_run_1"], length=70,
                        width=10,
                        thickness=4.0,
                        weight=14.4,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, ["training_run_1"], length=70,
                        width=10,
                        thickness=4.0,
                        weight=14.4,
                        diameter=6.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_830', screw_class_node, ["training_run_1"], length=30,
                        width=13,
                        thickness=5.3,
                        weight=15.5,
                        diameter=8.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_840', screw_class_node, ["training_run_1"], length=40,
                        width=13,
                        thickness=5.3,
                        weight=15.5,
                        diameter=8.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_850', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=50,
                        weight=21.8,
                        diameter=8.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_860', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=60,
                        weight=25.0,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_870', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=70,
                        weight=28.2,
                        diameter=8.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1030', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=30,
                        weight=26.2,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1040', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=40,
                        weight=31.2,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1050', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=50,
                        weight=36.2,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1060', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=60,
                        weight=41.3,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1070', screw_class_node, ["training_run_1"],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=70,
                        weight=46.3,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_530', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=30,
                        weight=5.6,
                        diameter=5.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_540', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=40,
                        weight=7.1,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_550', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=50,
                        weight=8.6,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_560', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=60,
                        weight=10.2,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_570', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=70,
                        weight=11.8,
                        diameter=5.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_630', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=30,
                        weight=8.7,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_640', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=40,
                        weight=11.0,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_650', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=50,
                        weight=13.2,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_660', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=60,
                        weight=15.4,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=70,
                        weight=17.6,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_830', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=30,
                        weight=16.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_840', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=40,
                        weight=20.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_850', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=50,
                        weight=24.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_860', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=60,
                        weight=28.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_870', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=70,
                        weight=33.0,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1030', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=30,
                        weight=27.9,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1040', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=40,
                        weight=34.1,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1050', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=50,
                        weight=40.3,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1060', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=60,
                        weight=46.5,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1070', screw_class_node, ["training_run_1"],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=70,
                        weight=52.7,
                        diameter=10.0,
                        coating=False))

        self.owl.add_node(GenericNode(node_id="row_1", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=0.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_2", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=0.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_3", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=0.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_4", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=30.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_5", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=30.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_6", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=30.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_7", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=60.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_8", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=60.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_9", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=60.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_10", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=90.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_11", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=90.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_12", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Silicon_gripper"], angle=90.00, success=True))
        self.owl.add_node(GenericNode(node_id="row_13", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_14", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_15", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_16", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_17", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_18", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_19", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_20", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_21", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_22", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_23", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_24", node_class=test_case_class_node,
                                      connections=["screw_530", "Niryo", "Einkerbung_gripper"], angle=90.00,
                                      success=True))

        self.owl.add_node(GenericNode(node_id="model_1", node_class=Model, connections=["training_run_1"]))

        self.owl.add_node(GenericNode(node_id="Test Durchgang", node_class=Attribute,
                                      connections=["niryo_dataset_september_2024"], datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schrauben ID", node_class=Attribute,
                                      connections=["niryo_dataset_september_2024"], datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schraubentyp", node_class=Attribute,
                                      connections=["niryo_dataset_september_2024"], datatype="Enum/ Kategorisch"))
        self.owl.add_node(GenericNode(node_id="TestCase1", node_class=Attribute,
                                      connections=["screw_530", "Niryo", "Silicone_gripper"], angle=0, success=True))

    # self.owl.add_node(GenericNode(node_id=""))

    def get_questionnaire(self):
        return self.questionnaire

    def get_ontology(self):
        return self.owl
