import os

import tensorflow as tf
from tensorflow.keras import layers, models

# Create output directory
os.makedirs("models", exist_ok=True)
model_dir = "models/simple_cnn"

print("Creating a simple CNN model...")


# Define a simple CNN model
def create_simple_cnn():
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Create and compile the model
model = create_simple_cnn()

# Print model information
print("\nModel created successfully!")
print("- Model type: Simple CNN")
print(f"- Input shape: {model.input_shape}")
print(f"- Output shape: {model.output_shape}")

# Save the model
print(f"\nSaving model to {model_dir}...")
model.save(f"{model_dir}.keras")  # Using .keras extension for Keras 3

print("\nModel saved successfully!")
print(f"- Model saved at: {os.path.abspath(model_dir)}")
print("\nModel summary:")
model.summary()

# Test the model with random data
print("\nTesting the model with random data...")
test_input = tf.random.normal((1, 28, 28, 1))
prediction = model.predict(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {prediction.shape}")
print(f"Sample prediction (first 5 values): {prediction[0][:5]}")

print("\nModel is ready for use with EdgeFlow!")
