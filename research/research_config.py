from research.owl import *
from research.questionnaire import Questionnaire


class Initialization():
    def __init__(self, demo_mode=False):
        self.owl = Ontology()
        self.owl.create_demo_ontology()
        self.questionnaire = Questionnaire(demo_mode)

        # Create Class Ontology
        screw_class_node = GenericClass("Screw", class_connections=[[], []], explanation="A metal Screw")
        test_case_class_node = GenericClass("TestCase", class_connections=[[], []],
                                            explanation="A screw is given to a roboter arm. The roboter arm has to "
                                                        "lift the screw and place it in a given place on a given angle. It is saved in the form of rows")
        robot_arm_class_node = GenericClass("Robotarm", class_connections=[["Gripper"], ["has"]],
                                            explanation="A robot arm with different modules.")
        gripper_class_node = GenericClass("Gripper", class_connections=[["Robotarm"], ["partOf"]], explanation="")
        feature_class_node = GenericClass("Feature", class_connections=[["Dataset"], ["belongsTo"]],
                                          explanation="Feature of a Dataset with statistics.")
        task = GenericClass(node_class_id="Task",
                            class_connections=[["Model"], ["achievedBy"]],
                            explanation="The Task, that a model achieves.")

        globalExplanationRun = GenericClass(node_class_id="GlobalExplanationRun",
                                            class_connections=[["Dataset", "GlobalInsight", "Model"],
                                                               ["hasInput", "hasOutput", "hasInput"]],
                                            explanation="A global Explanation Run is an algorithm which generates insight on a model. Examples are Shapley or Lime")
        globalInsight = GenericClass(node_class_id="GlobalInsight",
                                     class_connections=[["Model", "Attribute"], ["explains", "basedOn"]],
                                     explanation="A global Insight explains a model globally and not based on a single prediction.")

        self.owl.add_node(GenericNode(node_id="Insight_1_1", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_klammertyp"], ["explains", "basedOn"]],
                                      shap_value=1))
        self.owl.add_node(GenericNode(node_id="Insight_1_2", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_laenge"], ["explains", "basedOn"]],
                                      shap_value=0.5))
        self.owl.add_node(GenericNode(node_id="Insight_1_3", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_Schrauben_id"], ["explains", "basedOn"]],
                                      shap_value=0))

        self.owl.add_node(GenericNode(node_id="GlobalExplanationRun_1", node_class=globalExplanationRun, connections=[
            ["niryo_dataset_september_2024", "Insight_1_1", "Insight_1_2", "Insight_1_3"],
            ["hasInput", "hasOutput", "hasInput", "hasInput"]]))

        self.owl.add_node(GenericNode(node_id="Insight_2_1", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_klammertyp"], ["explains", "basedOn"]],
                                      shap_value=0.9))
        self.owl.add_node(GenericNode(node_id="Insight_2_2", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_laenge"], ["explains", "basedOn"]],
                                      shap_value=0.4))
        self.owl.add_node(GenericNode(node_id="Insight_2_3", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_Schrauben_id"], ["explains", "basedOn"]],
                                      shap_value=0.01))

        self.owl.add_node(GenericNode(node_id="GlobalExplanationRun_2", node_class=globalExplanationRun, connections=[
            ["niryo_dataset_september_2024", "Insight_2_1", "Insight_2_2", "Insight_2_3"],
            ["hasInput", "hasOutput", "hasInput", "hasInput"]]))

        self.owl.add_node(GenericNode(node_id="Insight_3_1", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_klammertyp"], ["explains", "basedOn"]],
                                      shap_value=1.1))
        self.owl.add_node(GenericNode(node_id="Insight_3_2", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_laenge"], ["explains", "basedOn"]],
                                      shap_value=0.6))
        self.owl.add_node(GenericNode(node_id="Insight_3_3", node_class=globalInsight,
                                      connections=[["model_a23b", "attribute_Schrauben_id"], ["explains", "basedOn"]],
                                      shap_value=0))

        self.owl.add_node(GenericNode(node_id="GlobalExplanationRun_3", node_class=globalExplanationRun, connections=[
            ["niryo_dataset_september_2024", "Insight_3_1", "Insight_3_2", "Insight_3_3"],
            ["hasInput", "hasOutput", "hasInput", "hasInput"]]))

        self.owl.add_node(
            GenericNode(node_id="ScrewPlacement", node_class=task,
                        connections=[["model_a23b", "model_xT77", "model_p1b3", "model_qdk1"], ["achievedBy", "achievedBy", "achievedBy", "achievedBy"]],
                        usecase="This Task is part of a non-critical research experiment."))

        self.owl.add_node(Attribute(node_id="attribute_Schrauben_id",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM",
                                    valueDistribution="530, 540, 550, 560, 570", attributeName="Schrauben ID"))
        self.owl.add_node(Attribute(node_id="attribute_Schraubentyp",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM",
                                    valueDistribution="Sechskant, Zylinder", attributeName="Schraubentyp"))
        self.owl.add_node(Attribute(node_id="attribute_kopfbreite",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="Numeric", valueDistribution="8-24.65",
                                    valueAverage="14.06",
                                    attributeName="Kopfbreite in MM"))
        self.owl.add_node(Attribute(node_id="attribuite_Kopfdicke",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="Numeric", valueDistribution="3.5-10",
                                    attributeName="Kopfdicke in MM"))
        self.owl.add_node(Attribute(node_id="attribute_laenge",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="Numeric", valueDistribution="30-70",
                                    attributeName="Laenge in MM"))
        self.owl.add_node(Attribute(node_id="attribute_gewicht",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="Numeric",
                                    valueDistribution="5.1-52.7", attributeName="Gewicht (g)"))
        self.owl.add_node(Attribute(node_id="attribute_durchmesser",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="Numeric", valueDistribution="5-10",
                                    attributeName="Durchmesser (mm)"))
        self.owl.add_node(Attribute(node_id="attribute_beschichtung",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM", valueDistribution="Nein, JA",
                                    attributeName="Beschichtung"))
        self.owl.add_node(Attribute(node_id="attribute_klammertyp",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM",
                                    valueDistribution="Silicon, Einkerbung, Standard", attributeName="Klammer-Typ"))
        self.owl.add_node(Attribute(node_id="attribute_winkel",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM",
                                    valueDistribution="0, 30, 60, 90", attributeName="Winkel (in Grad)"))
        self.owl.add_node(Attribute(node_id="attribute_label",
                                    connections=[['preprocessing_niryo', 'niryo_dataset_september_2024'],
                                                 ["usedBy", "partOf"]], datatype="ENUM",
                                    valueDistribution="True, False", attributeName="Erfolgreich (Ja/Nein)",
                                    information="Filled in based on the result of the test"))

        self.owl.add_node(GenericNode("Niryo", robot_arm_class_node, connections=[["training_run_1"], ["usedIn"]]))
        self.owl.add_node(
            GenericNode('Silicon_gripper', gripper_class_node, connections=[["Niryo"], ["belongsTo"]], typ="silicone"))
        self.owl.add_node(
            GenericNode('Notch_gripper', gripper_class_node, connections=[["Niryo"], ["belongsTo"]], typ="notch"))
        self.owl.add_node(
            GenericNode('Einkerbung_gripper', gripper_class_node, connections=[["Niryo"], ["belongsTo"]],
                        typ="einkerbung"))
        self.owl.add_node(
            GenericNode('screw_530', screw_class_node, [["training_run_1"], ["usedIn"]], length=30,
                        width=8,
                        thickness=3.5,
                        weight=5.1,
                        diameter=5.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_540', screw_class_node, [["training_run_1"], ["usedIn"]], length=40,
                        width=8,
                        thickness=3.5,
                        weight=6.4,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_550', screw_class_node, [["training_run_1"], ["usedIn"]], length=50,
                        width=8,
                        thickness=3.5,
                        weight=7.6,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_560', screw_class_node, [["training_run_1"], ["usedIn"]], length=60,
                        width=8,
                        thickness=3.5,
                        weight=8.8,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_570', screw_class_node, [["training_run_1"], ["usedIn"]], length=70,
                        width=8,
                        thickness=3.5,
                        weight=10.2,
                        diameter=5.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_630', screw_class_node, [["training_run_1"], ["usedIn"]], length=30,
                        width=10,
                        thickness=4.0,
                        weight=7.5,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_640', screw_class_node, [["training_run_1"], ["usedIn"]], length=40,
                        width=10,
                        thickness=4.0,
                        weight=9.2,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_650', screw_class_node, [["training_run_1"], ["usedIn"]], length=50,
                        width=10,
                        thickness=4.0,
                        weight=11.0,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_660', screw_class_node, [["training_run_1"], ["usedIn"]], length=60,
                        width=10,
                        thickness=4.0,
                        weight=12.7,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, [["training_run_1"], ["usedIn"]], length=70,
                        width=10,
                        thickness=4.0,
                        weight=14.4,
                        diameter=6.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, [["training_run_1"], ["usedIn"]], length=70,
                        width=10,
                        thickness=4.0,
                        weight=14.4,
                        diameter=6.0,
                        type="Sechskant",
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_830', screw_class_node, [["training_run_1"], ["usedIn"]], length=30,
                        width=13,
                        thickness=5.3,
                        weight=15.5,
                        diameter=8.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_840', screw_class_node, [["training_run_1"], ["usedIn"]], length=40,
                        width=13,
                        thickness=5.3,
                        weight=15.5,
                        diameter=8.0,
                        type="Sechskant",
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_850', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=50,
                        weight=21.8,
                        diameter=8.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_860', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=60,
                        weight=25.0,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_870', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=13,
                        thickness=5.3,
                        length=70,
                        weight=28.2,
                        diameter=8.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1030', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=30,
                        weight=26.2,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1040', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=40,
                        weight=31.2,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1050', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=50,
                        weight=36.2,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1060', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=60,
                        weight=41.3,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1070', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Sechskant",
                        width=17,
                        thickness=6.4,
                        length=70,
                        weight=46.3,
                        diameter=10.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_530', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=30,
                        weight=5.6,
                        diameter=5.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_540', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=40,
                        weight=7.1,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_550', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=50,
                        weight=8.6,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_560', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=60,
                        weight=10.2,
                        diameter=5.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_570', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=8.5,
                        thickness=5.0,
                        length=70,
                        weight=11.8,
                        diameter=5.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_630', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=30,
                        weight=8.7,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_640', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=40,
                        weight=11.0,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_650', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=50,
                        weight=13.2,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_660', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=60,
                        weight=15.4,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_670', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=10.0,
                        thickness=6.0,
                        length=70,
                        weight=17.6,
                        diameter=6.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_830', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=30,
                        weight=16.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_840', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=40,
                        weight=20.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_850', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=50,
                        weight=24.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_860', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=60,
                        weight=28.9,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_870', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=13.0,
                        thickness=8.0,
                        length=70,
                        weight=33.0,
                        diameter=8.0,
                        coating=False))
        self.owl.add_node(
            GenericNode('screw_1030', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=30,
                        weight=27.9,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1040', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=40,
                        weight=34.1,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1050', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=50,
                        weight=40.3,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1060', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=60,
                        weight=46.5,
                        diameter=10.0,
                        coating=True))
        self.owl.add_node(
            GenericNode('screw_1070', screw_class_node, [["training_run_1"], ["usedIn"]],
                        type="Zylinder",
                        width=16.0,
                        thickness=10.0,
                        length=70,
                        weight=52.7,
                        diameter=10.0,
                        coating=False))

        self.owl.add_node(GenericNode(node_id="row_1", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_2", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_3", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_4", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_5", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_6", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_7", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_8", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_9", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_10", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_11", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_12", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_13", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_14", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_15", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_16", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_17", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_18", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=30.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_19", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_20", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_21", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=60.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_22", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_23", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))
        self.owl.add_node(GenericNode(node_id="row_24", node_class=test_case_class_node,
                                      connections=[["screw_530", "Niryo", "Einkerbung_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=90.00,
                                      success=True))

        # self.owl.add_node(
        #   GenericNode(node_id="model_a23b", node_class=Model, connections=[["training_run_1"], ["usedIn"]]))

        self.owl.add_node(
            GenericNode(node_id="model_xT77", node_class=Model, algorithm="Neural Net",
                        connections=[["niryo_dataset_september_2024", "ScrewPlacement"], ["used", "achieves"]],
                        accuracy=0.90))
        self.owl.add_node(
            GenericNode(node_id="model_p1b3", node_class=Model, algorithm="Naive Bayes",
                        connections=[["niryo_dataset_september_2024", "ScrewPlacement"], ["used", "achieves"]],
                        accuracy=0.90))
        self.owl.add_node(
            GenericNode(node_id="model_qdk1", node_class=Model, algorithm="Decision Tree",
                        connections=[["niryo_dataset_september_2024", "ScrewPlacement"], ["used", "achieves"]],
                        accuracy=0.92))
        self.owl.add_node(GenericNode(node_id="Test Durchgang", node_class=Attribute,
                                      connections=[["niryo_dataset_september_2024"], ["usedIn"]],
                                      datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schrauben ID", node_class=Attribute,
                                      connections=[["niryo_dataset_september_2024"], ["usedIn"]],
                                      datatype="Numerisch/ ID"))
        self.owl.add_node(GenericNode(node_id="Schraubentyp", node_class=Attribute,
                                      connections=[["niryo_dataset_september_2024"], ["usedIn"]],
                                      datatype="Enum/ Kategorisch"))
        self.owl.add_node(GenericNode(node_id="TestCase1", node_class=Attribute,
                                      connections=[["screw_530", "Niryo", "Silicon_gripper"],
                                                   ["usedScrew", "usedRobot", "usedGripper"]], angle=0, success=True))

    # self.owl.add_node(GenericNode(node_id=""))

    def get_questionnaire(self):
        return self.questionnaire

    def get_ontology(self):
        return self.owl
