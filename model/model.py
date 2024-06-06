import tensorflow as tf
from keras._tf_keras.keras.layers import Dense, Input, BatchNormalization, Dropout
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('starrcitizen.csv', sep=';', header=0, names=['Model', 'Manufacturer', 'Focus', 'Length', 'Beam', 'Height', 'Airworthy', 'Size', 'Mass', 'Cargo Capacity', 'SCM Speed', 'Afterburner Speed', 'Min Crew', 'Max Crew', 'Price'])

data.drop(columns=['Model', 'Min Crew', 'Max Crew', 'Airworthy'], inplace=True)

# Encode categorical variables
categorical_encoders = ['Manufacturer', 'Focus', 'Size']
label_encoders = {}
for col in categorical_encoders:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Ensure all data types are numeric
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = label_encoders[col].fit_transform(data[col])

# Feature scaling
scaler = StandardScaler()
numerical_cols = ['Length', 'Beam', 'Height', 'Mass', 'Cargo Capacity', 'SCM Speed', 'Afterburner Speed']
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Separate features and target variable
X = data.drop(columns='Price', axis=1)
y = data['Price']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mae'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(X_train, y_train, callbacks=[early_stopping, model_checkpoint, reduce_lr], validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)

# Save the entire model
model.save('model.h5')

# Plot learning curves
def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    plt.title('Learning Curves')
    plt.tight_layout()
    plt.show()

# Call the plot function
plot_learning_curves(history)

# Function to preprocess user input
def preprocess_input(input_data):
    for col, encoder in label_encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]
    input_data = pd.DataFrame([input_data])
    input_data = input_data.astype(float)
    return input_data

# Prediction function
def predict():
    print("Enter the details for the following features: ")
    manufacturer_input = input("Manufacturer: ")
    focus_input = input("Focus: ")
    length_input = input("Length: ")
    beam_input = input("Beam: ")
    height_input = input("Height: ")
    size_input = input("Size: ")
    mass_input = input("Mass: ")
    cargo_capacity_input = input("Cargo Capacity: ")
    scm_speed_input = input("SCM Speed: ")
    afterburner_speed_input = input("Afterburner Speed: ")

    try:
        input_data = {
            'Manufacturer': manufacturer_input,
            'Focus': focus_input,
            'Length': float(length_input),
            'Beam': float(beam_input),
            'Height': float(height_input),
            'Size': size_input,
            'Mass': float(mass_input),
            'Cargo Capacity': float(cargo_capacity_input),
            'SCM Speed': float(scm_speed_input),
            'Afterburner Speed': float(afterburner_speed_input),
        }

        # Preprocess input data
        preprocessed_data = preprocess_input(input_data)

        # Scale numerical features
        preprocessed_data[numerical_cols] = scaler.transform(preprocessed_data[numerical_cols])

        # Make prediction
        model = load_model('best_model.keras')
        prediction = model.predict(preprocessed_data)
        print(f"Predicted Price: {prediction[0][0]:.2f}")
    except Exception as e:
        print(f"Invalid input: {e}")
