"""
evaluate.py
Evaluasi model yang sudah dilatih terhadap test set: accuracy, precision,
recall, F1-score, confusion matrix (gambar), dan classification report
(teks). Semua hasil disimpan di folder reports/ supaya gampang ditempel
ke laporan/skripsi.

Jalankan dari root proyek:
    python -m src.evaluate
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

from . import config
from .dataset import load_datasets


def main(model_path: str = None):
    model_path = model_path or config.BEST_CHECKPOINT_PATH
    print(f">> Memuat model dari: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(">> Memuat test set ...")
    _, _, test_ds = load_datasets()

    y_true = []
    y_prob = []
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x, verbose=0).flatten()
        y_prob.extend(preds.tolist())
        y_true.extend(batch_y.numpy().flatten().tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, digits=4
    )
    print(report)

    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    with open(os.path.join(config.REPORTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nROC-AUC: {auc:.4f}\n")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - CIFAKE Test Set")
    fig.tight_layout()
    fig.savefig(os.path.join(config.REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    print(">> classification_report.txt & confusion_matrix.png tersimpan di reports/")


if __name__ == "__main__":
    main()
