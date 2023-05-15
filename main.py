import prediction
import explanation
import kgconnector

if __name__ == '__main__':
    result = prediction.predict([[0, 50.68, 0, 0, 1, 0]])
    print(result)

    kgconnector.save_prediction()
