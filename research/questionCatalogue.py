class Questionnaire:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

    def get_question(self, key):
        """Returns the value associated with the given key in dict1."""
        return self.questions.get(key, None)  # Returns None if the key is not found

    def get_answer(self, key):
        """Returns the value associated with the given key in dict2."""
        return self.answers.get(key, None)  # Returns None if the key is not found

    def get_question_answer_pair(self, key):
        """Returns a tuple of values from both dict1 and dict2 for the given key."""
        return self.questions.get(key, None), self.answers.get(key, None)

questions = {
    1:"How many entries does data set DataSet_58ddb600?",
    2:"How many attributes does data set DataSet_58ddb600?",
    3:"What are the datatypes of the features of DataSet_58ddb600?",
    4:"What are the value distributions of Feature_87176016?",
    5:"In which environment (location, company, system, etc.) was the data set collected?",
    6:"When was the data collected?",
    8:"How was the label filled in the training data?",
    9:"Which algorithm was used?",
    10:"How was the label filled in the training data?"
    }

answers = {
    1:"Data set DataSet_58ddb600-f0dc-47df-bc4c-a88c00950fab contains 300 entries.",
    2:"The data set DataSet_58ddb600-f0dc-47df-bc4c-a88c00950fab has 4 attributes.",
    3:"The dataset DataSet_58ddb600-f0dc-47df-bc4c-a88c00950fab has at least one feature, Feature_87176016, which has a datatype of numerical. Information about other features' datatypes is not provided.",
    4:"Based on the provided attributes for Feature_87176016-fc01-46bb-80a8-04ad05df69f8, we can infer the following about its value distribution: Mean (μ): 3.003 Standard Deviation (σ): 0.310 Minimum Value: 2.45 Maximum Value: 4.91Given that the feature is numerical and assuming it follows a normal distribution (which is a common assumption for many numerical features unless specified otherwise), the distribution can be described as:Approximately Normal Distribution: The values are likely to be symmetrically distributed around the mean (3.003) with most values falling within a range defined by the standard deviation.",
    5:"The data set was collected in the environment of RWU (Ravensburg-Weingarten University of Applied Sciences) and is associated with the FestoSystem domain.",
    6:"The data set was collected during the fourth quarter (Q4) of 2023.",
    8:"""The label in the training data, identified as `Label_fa5649f9-0f84-4723-845a-6f5bd43d5b9f`, corresponds to the attribute named Result (`Attribute_a11be75b-515b-420f-9aa1-3338e19d3f75`). This attribute was part of the preprocessing step carried out by `Preprocessing_b9875fe0-fe87-4af2-b5ac-c8b9e64beb59`.
During the preprocessing, the data was read from a CSV file named "sparqlResult.csv". The preprocessing script dropped the columns 'experiment' and 'cylinder' from the dataset, leaving the remaining columns, including 'result', which was used as the label for training the decision tree classifier.
Here is the relevant part of the preprocessing script:
```python
df = pd.read_csv("sparqlResult.csv")
df.drop('experiment', axis='columns', inplace=True)
df.drop('cylinder', axis='columns', inplace=True)
X = df.drop('result', axis=1)
y = df['result']
```
In this script:
- `X` represents the features used for training.
- `y` represents the label, which is the 'result' column.
The 'result' column was used as the label to train the decision tree classifier (`DecisionTreeClassifier`). The classifier was then trained using this label to predict outcomes based on the features provided in the dataset.
""",
    9:"""The algorithm used is the Decision Tree Classifier from the scikit-learn library. This is evident from the following lines in the provided Python preprocessing script:
```python
from sklearn.tree import DecisionTreeClassifier
...
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X, y)
```
""",
    10:"""The label in the training data, identified as `Label_fa5649f9-0f84-4723-845a-6f5bd43d5b9f`, corresponds to the attribute named "Result" (`Attribute_a11be75b-515b-420f-9aa1-3338e19d3f75`). This attribute was part of the preprocessing step carried out by `Preprocessing_b9875fe0-fe87-4af2-b5ac-c8b9e64beb59`.
During the preprocessing, the data was read from a CSV file named "sparqlResult.csv". The preprocessing script dropped the columns 'experiment' and 'cylinder' from the dataset, leaving the remaining columns, including 'result', which was used as the label for training the decision tree classifier.
Here is the relevant part of the preprocessing script:
```python
df = pd.read_csv("sparqlResult.csv")
df.drop('experiment', axis='columns', inplace=True)
df.drop('cylinder', axis='columns', inplace=True)
X = df.drop('result', axis=1)
y = df['result']
```
In this script:
- `X` represents the features used for training.
- `y` represents the label, which is the 'result' column.
The 'result' column was used as the label to train the decision tree classifier (`DecisionTreeClassifier`). The classifier was then trained using this label to predict outcomes based on the features provided in the dataset.
"""

}