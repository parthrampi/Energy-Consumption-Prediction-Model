from src.preprocess import load_data, preprocess_data
from src.train import train_model
from src.predict import predict_power

# Load and preprocess
df = load_data("data/energy_data.csv")
X, y = preprocess_data(df)

# Train model
train_model(X, y)

# Predict example
pred = predict_power(220, 1.6, 32)
print(f"Predicted Power: {pred}")
