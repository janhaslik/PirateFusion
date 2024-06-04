import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv('starrcitizen.csv', sep=';', header=0, names=['Model', 'Manufacturer', 'Focus', 'Length', 'Beam', 'Height', 'Airworthy', 'Size', 'Mass', 'Cargo Capacity', 'SCM Speed', 'Afterburner Speed', 'Min Crew', 'Max Crew', 'Price'])

# Encode categorical variables
categorical_encoders = ['Model', 'Manufacturer', 'Focus', 'Airworthy', 'Size']
label_encoders = {}
for col in categorical_encoders:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Ensure all data types are numeric
for col in data.columns:
    if data[col].dtype == 'object':
        print(f"Column {col} has non-numeric data")
        data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target variable
X = data.drop(columns='Price', axis=1)
y = data['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error on test set: {mae:.2f}")

# Function to preprocess user input
def preprocess_input(input_data):
    for col, encoder in label_encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]
    input_data = pd.DataFrame([input_data])
    input_data = input_data.astype(float)
    return input_data

# Prediction loop
def predict():
    print("Enter the details for the following features: ")
    model_input = input("Model: ")
    manufacturer_input = input("Manufacturer: ")
    focus_input = input("Focus: ")
    length_input = input("Length: ")
    beam_input = input("Beam: ")
    height_input = input("Height: ")
    airworthy_input = input("Airworthy: ")
    size_input = input("Size: ")
    mass_input = input("Mass: ")
    cargo_capacity_input = input("Cargo Capacity: ")
    scm_speed_input = input("SCM Speed: ")
    afterburner_speed_input = input("Afterburner Speed: ")
    min_crew_input = input("Min Crew: ")
    max_crew_input = input("Max Crew: ")

    try:
        input_data = {
            'Model': model_input,
            'Manufacturer': manufacturer_input,
            'Focus': focus_input,
            'Length': float(length_input),
            'Beam': float(beam_input),
            'Height': float(height_input),
            'Airworthy': airworthy_input,
            'Size': size_input,
            'Mass': float(mass_input),
            'Cargo Capacity': float(cargo_capacity_input),
            'SCM Speed': float(scm_speed_input),
            'Afterburner Speed': float(afterburner_speed_input),
            'Min Crew': float(min_crew_input),
            'Max Crew': float(max_crew_input)
        }
        preprocessed_data = preprocess_input(input_data)
        prediction = model.predict(preprocessed_data)
        print(f"Predicted Price: {prediction[0][0]:.2f}")
    except Exception as e:
        print(f"Invalid input: {e}")
