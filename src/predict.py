"""
predict.py
Coba model ke satu file gambar dari command line. Berguna buat demo cepat
sebelum bikin UI (Streamlit/mobile app).

Contoh pakai model Keras:
    python -m src.predict --image contoh.jpg

Contoh pakai model TFLite (yang nanti dipasang ke HP):
    python -m src.predict --image contoh.jpg --tflite models/cifake_cnn_quant.tflite
"""

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

from . import config


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def predict_keras(image_arr: np.ndarray, model_path: str) -> float:
    model = tf.keras.models.load_model(model_path)
    batch = np.expand_dims(image_arr, axis=0)
    prob = model.predict(batch, verbose=0)[0][0]
    return float(prob)


def predict_tflite(image_arr: np.ndarray, tflite_path: str) -> float:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    batch = np.expand_dims(image_arr, axis=0).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], batch)
    interpreter.invoke()
    prob = interpreter.get_tensor(output_details[0]["index"])[0][0]
    return float(prob)


def interpret(prob_real: float) -> str:
    # Ingat: label index 1 = REAL, 0 = FAKE (lihat config.CLASS_NAMES)
    label = "REAL (foto asli)" if prob_real >= 0.5 else "FAKE (dibuat AI)"
    confidence = prob_real if prob_real >= 0.5 else 1 - prob_real
    return f"{label}  |  confidence: {confidence * 100:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="path ke file gambar")
    parser.add_argument("--model", default=config.BEST_CHECKPOINT_PATH,
                         help="path ke model .keras (dipakai kalau --tflite tidak diisi)")
    parser.add_argument("--tflite", default=None,
                         help="path ke model .tflite (opsional, override --model)")
    args = parser.parse_args()

    image_arr = load_image(args.image)

    if args.tflite:
        prob_real = predict_tflite(image_arr, args.tflite)
    else:
        prob_real = predict_keras(image_arr, args.model)

    print(interpret(prob_real))


if __name__ == "__main__":
    main()
