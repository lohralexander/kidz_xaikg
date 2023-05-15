import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

import kgconnector
import processing as pr

if __name__ == '__main__':
    training_data = kgconnector.get_training_data()
    df = pd.DataFrame.from_dict(training_data['results']['bindings'], orient='columns')
    df = df.applymap(pr.get_values_from_json)
    # https://stackoverflow.com/questions/51306491/applying-a-method-to-a-few-selected-columns-in-a-pandas-dataframe
    columns_to_clean = ['experiment', 'cylinder', 'material', 'bottom']
    df[columns_to_clean] = df[columns_to_clean].applymap(pr.remove_owl_uri)

    df.drop('experiment', axis='columns', inplace=True)
    df.drop('cylinder', axis='columns', inplace=True)
    X = df.drop('result', axis=1)
    y = df['result']
    one_hot = pd.get_dummies(df['bottom'])
    dfjoin = df.join(one_hot)
    df1 = dfjoin.drop('bottom', axis=1)
    df = df1.drop('material', axis=1)
    df.rename(columns={'http://www.semanticweb.org/kidz/festo#concave025': 'concave025',
                       'http://www.semanticweb.org/kidz/festo#concave05': 'concave05',
                       'http://www.semanticweb.org/kidz/festo#concave075': 'concave075',
                       'http://www.semanticweb.org/kidz/festo#even': 'even'}, inplace=True)
    X = df.drop('result', axis=1)
    y = df['result']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        stratify=y)
    gbc = GradientBoostingClassifier(n_estimators=50, min_samples_split=10, max_depth=None, learning_rate=0.1,
                                     random_state=41).fit(X_train, y_train)
    pickle.dump(gbc, open('test.pickle', 'wb'))
