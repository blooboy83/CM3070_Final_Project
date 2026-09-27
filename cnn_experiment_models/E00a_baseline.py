# type: ignore
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import keras
from keras import layers, models
import pickle
import matplotlib.pyplot as plt

# Path to pre-processed data
SAVE_DIR = "./processed_data/"

# Load the saved arrays
X_train = np.load(os.path.join(SAVE_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(SAVE_DIR, 'y_train.npy'))
X_val = np.load(os.path.join(SAVE_DIR, 'X_val.npy'))
y_val = np.load(os.path.join(SAVE_DIR, 'y_val.npy'))
X_test = np.load(os.path.join(SAVE_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(SAVE_DIR, 'y_test.npy'))

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Combine train and val 
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

# Add channel dimension (grayscale)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Convert labels to categorical (binary)
num_classes = 2
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

data is (224, 224, 1) - grayscale images
model = models.Sequential([
    
    layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(224, 224, 1)), 
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.Flatten(),

    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  
])

model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    epochs=15,
                    batch_size=64,
                    validation_split=0.2,
                    verbose=2)

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Get predictions for metrics
y_pred_probs = model.predict(X_test, verbose=0)

# Convert one-hot back to labels for metrics
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Core metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)  # sensitivity
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
f1 = f1_score(y_true, y_pred, zero_division=0)
auc = roc_auc_score(y_true, y_pred_probs[:, 1])

print("\n" + "="*70)
print("BASELINE MODEL - COMPLETE RESULTS")
print("="*70)

print("\nSTANDARD TEST METRICS:")
print(f"   Test Accuracy: {test_accuracy:.4f}  ({test_accuracy*100:.2f}%)")
print(f"   Test Loss:     {test_loss:.4f}")

print("\nCLASSIFICATION METRICS:")
print(f"   Accuracy:    {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"   Precision:   {precision:.4f}  ({precision*100:.2f}%)")
print(f"   Recall:      {recall:.4f}  ({recall*100:.2f}%)  ← Sensitivity")
print(f"   Specificity: {specificity:.4f}  ({specificity*100:.2f}%)")
print(f"   F1 Score:    {f1:.4f}  ({f1*100:.2f}%)")
print(f"   AUC:         {auc:.4f}  ({auc*100:.2f}%)")

print("\nCONFUSION MATRIX:")
print(f"                 Predicted")
print(f"                 Benign   Malignant")
print(f"   Actual Benign    {tn:>5}     {fp:>5}")
print(f"          Malignant  {fn:>5}     {tp:>5}")


print(f"Experiment ID:\tE00_Baseline")
print(f"Test Accuracy:\t{test_accuracy:.4f}")
print(f"Test Loss:\t{test_loss:.4f}")
print(f"Accuracy:\t{accuracy:.4f}")
print(f"Precision:\t{precision:.4f}")
print(f"Recall:\t\t{recall:.4f}")
print(f"Specificity:\t{specificity:.4f}")
print(f"F1 Score:\t{f1:.4f}")
print(f"AUC:\t\t{auc:.4f}")
print(f"TP:\t\t{tp}")
print(f"TN:\t\t{tn}")
print(f"FP:\t\t{fp}")
print(f"FN:\t\t{fn}")


model.save('baseline.keras')
print("Saved as 'baseline.keras'")

with open('training_history_E00.pkl', 'wb') as f:
    pickle.dump(history.history, f)
print("Training history saved")