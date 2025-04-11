import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
def load_data():
    # Sample dataset
    data = {
        "Rooms": [1, 2, 3, 4, 5],
        "Area": [400, 600, 800, 1000, 1200],
        "Price": [200000, 250000, 300000, 400000, 500000],
    }
    return pd.DataFrame(data)

# Train the model
def train_model():
    df = load_data()
    X = df[["Rooms", "Area"]]
    y = df["Price"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

# Predict house prices
def predict(model, rooms, area):
    return model.predict([[rooms, area]])

if __name__ == "__main__":
    print("Training the model...")
    model = train_model()

    print("Making a prediction for a house with 3 rooms and 850 sq ft area...")
    price = predict(model, 3, 850)
    print(f"Predicted Price: ${price[0]:,.2f}")
