# code based on https://machinelearningmastery.com/mcnemars-test-for-machine-learning/

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

SAVE_DIR = "./augmented_none/"
CNN_MODEL_PATH = "./models/E06e_5_layers_no_aug_seed_70.keras"
VGG_MODEL_PATH = "./models/ET01a_VGG16_seed_70.keras"
CNN_THRESHOLD = 0.51
VGG_THRESHOLD = 0.45
PLOTS_DIR = "./plots/"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Load test data
X_test = np.load(os.path.join(SAVE_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(SAVE_DIR, 'y_test.npy'))
print(f"Raw X_test shape: {X_test.shape}")

# Prepare CNN input
X_test_cnn = X_test
while X_test_cnn.ndim < 4:
    X_test_cnn = X_test_cnn[..., np.newaxis]
if X_test_cnn.shape[-1] != 1:
    X_test_cnn = X_test_cnn[..., :1]
print(f"CNN input shape: {X_test_cnn.shape}")

# Prepare VGG input
X_test_vgg = X_test
while X_test_vgg.ndim < 4:
    X_test_vgg = X_test_vgg[..., np.newaxis]
if X_test_vgg.shape[-1] == 1:
    X_test_vgg = np.repeat(X_test_vgg, 3, axis=-1)
print(f"VGG input shape: {X_test_vgg.shape}")


# Predictions
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)[:, 1]
cnn_pred = (cnn_probs >= CNN_THRESHOLD).astype(int)

vgg_model = tf.keras.models.load_model(VGG_MODEL_PATH)
vgg_probs = vgg_model.predict(X_test_vgg, verbose=0)[:, 1]
vgg_pred = (vgg_probs >= VGG_THRESHOLD).astype(int)


# McNemar test
a = int(np.sum((cnn_pred == y_test) & (vgg_pred == y_test)))
b = int(np.sum((cnn_pred == y_test) & (vgg_pred != y_test)))
c = int(np.sum((cnn_pred != y_test) & (vgg_pred == y_test)))
d = int(np.sum((cnn_pred != y_test) & (vgg_pred != y_test)))
table = [[a, b], [c, d]]

result = mcnemar(table, exact=(b + c) < 25, correction=False)
print(f"\nMcNemar: chi2={result.statistic:.4f}, p={result.pvalue:.4f}")
print(f"  Both correct: {a}")
print(f"  CNN only: {b}")
print(f"  VGG only: {c}")
print(f"  Both wrong: {d}")


# FIGURE 3: ROC curves
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_probs)
fpr_vgg, tpr_vgg, _ = roc_curve(y_test, vgg_probs)
auc_cnn = auc(fpr_cnn, tpr_cnn)
auc_vgg = auc(fpr_vgg, tpr_vgg)

plt.figure(figsize=(7, 6))
plt.plot(fpr_cnn, tpr_cnn, label=f"Custom CNN (AUC = {auc_cnn:.3f})", linewidth=2)
plt.plot(fpr_vgg, tpr_vgg, label=f"VGG16 (AUC = {auc_vgg:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4, label="Chance")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig3_roc_seed70.png"), dpi=150)
plt.close()
print(f"\nSaved: fig3_roc_seed70.png")


# FIGURE 4: Confusion matrices
cm_cnn = confusion_matrix(y_test, cnn_pred, labels=[0, 1])
cm_vgg = confusion_matrix(y_test, vgg_pred, labels=[0, 1])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'], ax=axes[0])
axes[0].set_title("Custom CNN")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(cm_vgg, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'], ax=axes[1])
axes[1].set_title("VGG16")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig4_confusion_matrices_seed70.png"), dpi=150)
plt.close()
print(f"Saved: fig4_confusion_matrices_seed70.png")


# FIGURE 5: Threshold sweep
thresholds = np.arange(0.05, 0.96, 0.01)
recalls, specificities = [], []
for t in thresholds:
    pred = (cnn_probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    recalls.append(tp / (tp + fn) if (tp + fn) else 0)
    specificities.append(tn / (tn + fp) if (tn + fp) else 0)

plt.figure(figsize=(7, 5))
plt.plot(thresholds, recalls, label="Recall / Sensitivity", linewidth=2)
plt.plot(thresholds, specificities, label="Specificity", linewidth=2)
plt.axvline(CNN_THRESHOLD, color='red', linestyle='--',
            label=f"Selected threshold ({CNN_THRESHOLD:.2f})")
plt.xlabel("Classification threshold")
plt.ylabel("Metric value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig5_threshold_sweep_seed70.png"), dpi=150)
plt.close()
print(f"Saved: fig5_threshold_sweep_seed70.png")

print("\nDone. All figures saved to", PLOTS_DIR)