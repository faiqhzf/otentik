"""
config.py
Semua konstanta & hyperparameter proyek ada di sini.

Proyek ini adalah LANJUTAN dari cifake-detector: masalah yang ditemukan
di versi sebelumnya (model CIFAKE gagal generalisasi ke foto wajah/selfie
manusia, karena CIFAR-10 tidak punya kelas manusia sama sekali) diatasi
dengan retraining di dataset wajah asli vs AI, memakai bobot model CIFAKE
sebagai titik awal (transfer learning), bukan dari nol.
"""

import os

# ---- Path dataset ----
# Dataset: "140k Real and Fake Faces" (xhlulu, Kaggle)
# https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
# 70.000 wajah asli dari FFHQ (Flickr-Faces-HQ, Nvidia) + 70.000 wajah
# buatan StyleGAN. Sudah dibagi train/valid/test oleh pembuat dataset.
#
# Struktur folder yang diharapkan setelah download & extract (biasanya ada
# folder pembungkus "real_vs_fake/real-vs-fake/" -- notebook Colab akan
# otomatis mendeteksi & merapikannya, jadi tidak perlu diubah manual):
#
# data/
#   train/
#     real/
#     fake/
#   valid/
#     real/
#     fake/
#   test/
#     real/
#     fake/

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "valid")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Model CIFAKE lama (dilatih di tugas sebelumnya) dipakai sebagai warm-start.
# Upload file ini saat diminta notebook Colab, atau taruh manual di path ini
# kalau jalan di lokal/VS Code.
PRETRAINED_CIFAKE_PATH = os.path.join(MODELS_DIR, "cifake_cnn_best.keras")

KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "face_cnn.keras")
BEST_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "face_cnn_best.keras")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "face_cnn.tflite")
TFLITE_QUANT_MODEL_PATH = os.path.join(MODELS_DIR, "face_cnn_quant.tflite")

# ---- Gambar & data ----
# Wajah butuh resolusi lebih tinggi dari CIFAKE (32x32) supaya artefak
# GAN/diffusion yang halus tidak hilang saat resize. 128x128 dipilih
# sebagai titik tengah: cukup detail, masih ringan & cepat di GPU Colab.
# (Arsitektur modelnya pakai GlobalAveragePooling2D, jadi resolusi input
# BOLEH beda dari model lama -- bobot lama tetap bisa di-load, sudah
# diverifikasi tidak ada mismatch shape.)
IMG_SIZE = (128, 128)
CHANNELS = 3
BATCH_SIZE = 32
SEED = 42

# Nama kelas -> index. image_dataset_from_directory akan mengurutkan
# folder secara alfabetis, jadi fake=0, real=1.
CLASS_NAMES = ["fake", "real"]

# ---- Training ----
EPOCHS = 20
LEARNING_RATE = 3e-4  # lebih kecil dari training dari-nol (1e-3), karena warm-start
EARLY_STOPPING_PATIENCE = 4
FINE_TUNE_FROM_PRETRAINED = True  # set False untuk training dari nol (tanpa warm-start)
