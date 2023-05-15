import pickle as p


def predict(data):
    model = p.load(open('test.pickle', 'rb'))
    return int(model.predict(data))
