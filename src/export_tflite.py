"""
export_tflite.py
Convert model .keras -> .tflite supaya bisa dipasang di aplikasi mobile
(Android/iOS) atau di alat yang resource-nya terbatas.

Menghasilkan 2 versi:
  1. cifake_cnn.tflite        -> float32 biasa, tanpa kompresi tambahan
  2. cifake_cnn_quant.tflite  -> dynamic range quantization (bobot di
     kompres ke int8), ukurannya jauh lebih kecil & inferensinya lebih
     cepat di HP, dengan penurunan akurasi yang biasanya kecil.

Jalankan dari root proyek:
    python -m src.export_tflite
"""

import os

import tensorflow as tf

from . import config


def convert(model_path: str, out_path: str, quantize: bool):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(out_path) / 1024
    print(f">> Tersimpan: {out_path}  ({size_kb:.1f} KB)")
    return size_kb


def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    keras_size_kb = os.path.getsize(config.BEST_CHECKPOINT_PATH) / 1024
    print(f"Ukuran model Keras asli: {keras_size_kb:.1f} KB")

    plain_kb = convert(config.BEST_CHECKPOINT_PATH, config.TFLITE_MODEL_PATH, quantize=False)
    quant_kb = convert(config.BEST_CHECKPOINT_PATH, config.TFLITE_QUANT_MODEL_PATH, quantize=True)

    print("\nRingkasan ukuran model:")
    print(f"  Keras (.keras)            : {keras_size_kb:8.1f} KB")
    print(f"  TFLite float32 (.tflite)  : {plain_kb:8.1f} KB")
    print(f"  TFLite quantized (.tflite): {quant_kb:8.1f} KB")
    print("\nCatatan: setelah kuantisasi, cek ulang akurasinya di test set")
    print("(lihat predict.py / evaluate.py) sebelum dipasang ke app -")
    print("kadang ada sedikit penurunan akurasi yang perlu dicek dulu.")


if __name__ == "__main__":
    main()
