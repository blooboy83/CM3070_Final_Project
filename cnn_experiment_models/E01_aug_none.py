#type: ignore

import json
import datetime
import random
import os
import pickle

import numpy as np
import tensorflow as tf
import keras
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# CONFIGURATION — edit this for each experiment

CONFIG = {
    "run_id": "E01_no_augmentation",
    "description": "Early stopped 3-block CNN, no BN, no dropout, Adam default, no augmentation",

    # Architecture
    "conv_blocks": 3,
    "channels": [32, 64, 64],
    "dense_units": 64,
    "use_batchnorm": False,
    "dropout_rate": 0.0,
    "use_global_avg_pool": True,   # True = GlobalAveragePooling2D, False = Flatten

    # Training
    "learning_rate": 0.001, # keras default 0.001
    "batch_size": 64,
    "epochs": 50,
    "patience": 10,

    # Data / imbalance 
    "class_weight": None,
}

# GLOBAL SETTINGS
SAVE_DIR = "./augmented_none/"
MODELS_DIR = "./models/"
HISTORIES_DIR = "./histories/"
PLOTS_DIR = "./plots/"
LOG_PATH = "./experiments_log.json"
SEED = 42

# REPRODUCIBILITY
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# DATA LOADING
def load_data(save_dir):
    """Load preprocessed arrays. Adds channel dimension if missing.

    Does NOT perform any preprocessing
    """
    X_train = np.load(os.path.join(save_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(save_dir, 'y_train.npy'))
    X_val   = np.load(os.path.join(save_dir, 'X_val.npy'))
    y_val   = np.load(os.path.join(save_dir, 'y_val.npy'))
    X_test  = np.load(os.path.join(save_dir, 'X_test.npy'))
    y_test  = np.load(os.path.join(save_dir, 'y_test.npy'))

    # Add channel dimension if not already present
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
        X_val   = X_val[..., np.newaxis]
        X_test  = X_test[..., np.newaxis]

    return X_train, y_train, X_val, y_val, X_test, y_test

# MODEL CONSTRUCTION
def build_model(config, input_shape=(224, 224, 1)):
    """Build a CNN according to the config."""
    model_layers = []

    for i in range(config["conv_blocks"]):
        channels = config["channels"][i]
        model_layers.append(
            layers.Conv2D(channels, (3, 3), activation='relu', padding='same')
        )
        if config["use_batchnorm"]:
            model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.MaxPooling2D(2, 2))
        if config["dropout_rate"] > 0:
            model_layers.append(layers.Dropout(config["dropout_rate"]))

    if config["use_global_avg_pool"]:
        model_layers.append(layers.GlobalAveragePooling2D())
    else:
        model_layers.append(layers.Flatten())

    model_layers.append(layers.Dense(config["dense_units"], activation='relu'))
    if config["dropout_rate"] > 0:
        model_layers.append(layers.Dropout(config["dropout_rate"]))
    model_layers.append(layers.Dense(2, activation='softmax'))

    model = models.Sequential(model_layers)
    model.build(input_shape=(None,) + input_shape)
    return model

# EVALUATION
def evaluate_model(model, X_test, y_test):
    """Compute the full metric set on the test set."""
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    results = {
        "accuracy":    float(accuracy_score(y_test, y_pred)),
        "precision":   float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_test, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "f1":          float(f1_score(y_test, y_pred, zero_division=0)),
        "auc":         float(roc_auc_score(y_test, y_pred_probs[:, 1])),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    return results, y_pred_probs, y_pred

# PLOTTING
def save_training_plots(history, run_id, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    auc_hist = history.history.get('auc')
    val_auc_hist = history.history.get('val_auc')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if auc_hist is not None and val_auc_hist is not None:
        epochs_range = range(1, len(auc_hist) + 1)
        plt.figure()
        plt.plot(epochs_range, auc_hist, 'bo-', label='Training AUC')
        plt.plot(epochs_range, val_auc_hist, 'b-', label='Validation AUC')
        plt.title(f'Training and Validation AUC — {run_id}')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{run_id}_auc.png"), dpi=120)
        plt.close()

    if loss is not None and val_loss is not None:
        epochs_range = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs_range, loss, 'bo-', label='Training loss')
        plt.plot(epochs_range, val_loss, 'b-', label='Validation loss')
        plt.title(f'Training and Validation Loss — {run_id}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{run_id}_loss.png"), dpi=120)
        plt.close()

# LOGGING
def log_run(config, results, history, log_path):
    """Append this run's config and results to the JSON log."""
    best_epoch = int(np.argmax(history.history['val_auc']) + 1)
    best_val_auc = float(np.max(history.history['val_auc']))
    epochs_run = len(history.history['val_auc'])

    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "run_id": config["run_id"],
        "description": config["description"],
        "config": config,
        "results": results,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "epochs_run": epochs_run,
    }

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []

    log.append(entry)
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    return entry

# MAIN
def main():
    set_seeds(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(HISTORIES_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print(f"\n=== Run: {CONFIG['run_id']} ===")
    print(f"Description: {CONFIG['description']}\n")

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(SAVE_DIR)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Diagnostics
    print("\n--- Data diagnostics ---")
    print(f"Train labels: {np.bincount(y_train)}")
    print(f"Val labels:   {np.bincount(y_val)}")
    print(f"Test labels:  {np.bincount(y_test)}")
    print(f"Train proportions: {np.bincount(y_train) / len(y_train)}")

    majority = int(np.argmax(np.bincount(y_train)))
    trivial_pred = np.full_like(y_test, majority)
    print(f"Trivial accuracy on test: {accuracy_score(y_test, trivial_pred):.4f}")
    
    # Encode labels
    num_classes = 2
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   num_classes)
    y_test_cat  = tf.keras.utils.to_categorical(y_test,  num_classes)

    # Build model
    input_shape = X_train.shape[1:]  # (224, 224, 1)
    model = build_model(CONFIG, input_shape=input_shape)
    model.summary()

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss='categorical_crossentropy',
        metrics=[
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
        ],
    )

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=CONFIG["patience"],
        restore_best_weights=True,
    )

    # Train
    print("\nStarting training...", flush=True)
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        verbose=1,
        callbacks=[early_stopping],
        class_weight=CONFIG["class_weight"],
    )
    print("\nTraining completed.", flush=True)

    # Evaluate
    results, y_pred_probs, y_pred = evaluate_model(model, X_test, y_test)

    # Save artifacts
    model.save(os.path.join(MODELS_DIR, f"{CONFIG['run_id']}.keras"))
    with open(os.path.join(HISTORIES_DIR, f"{CONFIG['run_id']}_history.pkl"), 'wb') as f:
        pickle.dump(history.history, f)
    save_training_plots(history, CONFIG["run_id"], PLOTS_DIR)

    # Log
    entry = log_run(CONFIG, results, history, LOG_PATH)

    # Report
    print("\n=== RESULTS ===")
    print(f"Run ID:        {CONFIG['run_id']}")
    print(f"Best epoch:    {entry['best_epoch']} / {entry['epochs_run']}")
    print(f"Best val AUC:  {entry['best_val_auc']:.4f}")
    print(f"Accuracy:      {results['accuracy']:.4f}")
    print(f"Precision:     {results['precision']:.4f}")
    print(f"Recall:        {results['recall']:.4f}  (Sensitivity)")
    print(f"Specificity:   {results['specificity']:.4f}")
    print(f"F1 Score:      {results['f1']:.4f}")
    print(f"AUC:           {results['auc']:.4f}")
    print(f"TP: {results['tp']}  TN: {results['tn']}  FP: {results['fp']}  FN: {results['fn']}")
    print(f"\nLogged to {LOG_PATH}")

if __name__ == "__main__":
    main()