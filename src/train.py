"""
train.py
Jalankan training dari root proyek dengan:

    python -m src.train

Beda dari versi CIFAKE: sebelum training, model di-warm-start dari bobot
model CIFAKE lama (models/cifake_cnn_best.keras) kalau file itu ada dan
config.FINE_TUNE_FROM_PRETRAINED=True. Arsitektur pakai GlobalAveragePooling2D
sehingga bobot tetap kompatibel walau resolusi input berbeda (32x32 -> 128x128).

Akan menghasilkan:
  - models/face_cnn_best.keras   (checkpoint terbaik berdasarkan val_loss)
  - models/face_cnn.keras        (model di epoch terakhir)
  - reports/training_history.png   (grafik akurasi & loss per epoch)
  - reports/training_history.csv   (angka mentahnya, buat lampiran laporan)
"""

import os
import csv

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa display (headless/server)
import matplotlib.pyplot as plt

from . import config
from .dataset import load_datasets
from .model import build_lightweight_cnn, compile_model


def plot_history(history, out_path):
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs_range, hist["loss"], label="train_loss")
    axes[0].plot(epochs_range, hist["val_loss"], label="val_loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, hist["accuracy"], label="train_accuracy")
    axes[1].plot(epochs_range, hist["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Akurasi per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_history_csv(history, out_path):
    hist = history.history
    keys = list(hist.keys())
    n_epochs = len(hist[keys[0]])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(n_epochs):
            writer.writerow([i + 1] + [hist[k][i] for k in keys])


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    print(">> Memuat dataset ...")
    train_ds, val_ds, test_ds = load_datasets()

    print(">> Membangun model ...")
    model = build_lightweight_cnn(input_shape=(*config.IMG_SIZE, config.CHANNELS))
    model = compile_model(model)

    if config.FINE_TUNE_FROM_PRETRAINED and os.path.exists(config.PRETRAINED_CIFAKE_PATH):
        print(f">> Ditemukan model CIFAKE lama di {config.PRETRAINED_CIFAKE_PATH}")
        print(">> Warm-start: memuat bobotnya sebagai titik awal (transfer learning) ...")
        try:
            model.load_weights(config.PRETRAINED_CIFAKE_PATH)
            print(">> Berhasil! Training akan lanjut dari bobot CIFAKE, bukan dari nol.")
        except Exception as e:
            print(f">> Gagal load bobot lama ({e}); lanjut training dari nol.")
    else:
        print(">> Tidak ada model lama / FINE_TUNE_FROM_PRETRAINED=False -> training dari nol.")

    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            config.BEST_CHECKPOINT_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(">> Mulai training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
    )

    model.save(config.KERAS_MODEL_PATH)
    print(f">> Model tersimpan di: {config.KERAS_MODEL_PATH}")
    print(f">> Checkpoint terbaik di: {config.BEST_CHECKPOINT_PATH}")

    plot_history(history, os.path.join(config.REPORTS_DIR, "training_history.png"))
    save_history_csv(history, os.path.join(config.REPORTS_DIR, "training_history.csv"))
    print(">> Grafik & log training tersimpan di folder reports/")

    print(">> Evaluasi cepat di test set ...")
    results = model.evaluate(test_ds, return_dict=True)
    for k, v in results.items():
        print(f"   {k}: {v:.4f}")


if __name__ == "__main__":
    main()
