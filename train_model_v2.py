import os
import json
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# SETTINGS
# ==========================================

DATA_DIR = "/Users/laxmi/ISL_Subtitle_Translator/data/processed_v2"

MODEL_DIR = "/Users/laxmi/ISL_Subtitle_Translator/models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "isl_model_v2.keras"
)

LABEL_PATH = os.path.join(
    MODEL_DIR,
    "labels_v2.json"
)

SEQUENCE_LENGTH = 30
FEATURES = 1662

EPOCHS = 40
BATCH_SIZE = 8

# ==========================================
# LOAD DATA
# ==========================================

print()
print("======================================")
print("        ISL V2 MODEL TRAINING")
print("======================================")
print()

X = []
y = []

labels = sorted([
    folder
    for folder in os.listdir(DATA_DIR)
    if os.path.isdir(
        os.path.join(DATA_DIR, folder)
    )
])

print("Found labels:", len(labels))
print()

for label in labels:

    label_path = os.path.join(
        DATA_DIR,
        label
    )

    files = [
        file
        for file in os.listdir(label_path)
        if file.endswith(".npy")
    ]

    print(
        f"Loading {label}: {len(files)} samples"
    )

    for file in files:

        file_path = os.path.join(
            label_path,
            file
        )

        sequence = np.load(file_path)

        # Verify shape
        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURES
        ):
            print(
                "Skipping wrong shape:",
                file,
                sequence.shape
            )
            continue

        X.append(sequence)
        y.append(label)

# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

print()
print("======================================")
print("DATASET INFORMATION")
print("======================================")
print()

print("X shape:", X.shape)
print("Number of samples:", len(X))
print("Number of classes:", len(labels))

# ==========================================
# LABEL ENCODING
# ==========================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

class_names = encoder.classes_.tolist()

# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

# Count samples in each class
unique_classes, class_counts = np.unique(
    y_encoded,
    return_counts=True
)

classes_with_enough_samples = set(
    unique_classes[class_counts >= 2]
)

train_indices = []
test_indices = []

for class_id in unique_classes:

    indices = np.where(
        y_encoded == class_id
    )[0]

    if class_id in classes_with_enough_samples:

        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.20,
            random_state=42
        )

        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    else:

        # If a class has only one sample,
        # keep it in training data.
        train_indices.extend(indices)

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]

print()
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ==========================================
# BUILD MODEL
# ==========================================

model = Sequential([
    
    Input(
        shape=(
            SEQUENCE_LENGTH,
            FEATURES
        )
    ),

    Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    ),

    Dropout(0.30),

    Bidirectional(
        LSTM(64)
    ),

    Dropout(0.30),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(0.20),

    Dense(
        len(class_names),
        activation="softmax"
    )
])

# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print()
print("======================================")
print("MODEL")
print("======================================")
print()

model.summary()

# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# ==========================================
# TRAIN
# ==========================================

print()
print("======================================")
print("STARTING TRAINING")
print("======================================")
print()

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_test,
        y_test
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

# ==========================================
# EVALUATE
# ==========================================

print()
print("======================================")
print("FINAL EVALUATION")
print("======================================")
print()

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print(
    f"Test Loss: {loss:.4f}"
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)

with open(
    LABEL_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )

print()
print("======================================")
print("TRAINING COMPLETE")
print("======================================")
print()

print("Model saved to:")
print(MODEL_PATH)

print()
print("Labels saved to:")
print(LABEL_PATH)

print()