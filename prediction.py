import pickle as p
from config import Config


def predict(data):
    model = p.load(open(Config.model_name, 'rb'))
    return int(model.predict(data))
