import pickle
import numpy as np

def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict_power(voltage, current, temperature):
    model = load_model()
    data = np.array([[voltage, current, temperature]])
    prediction = model.predict(data)

    return prediction[0]
