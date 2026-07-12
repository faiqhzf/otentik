"""
model.py
Arsitektur CNN ringan untuk klasifikasi biner REAL vs FAKE.

Desainnya terinspirasi dari pendekatan "lightweight CNN" yang dipakai
Lokner Ladjevic dkk. (2024, MDPI AI 5(3):1575-1593) untuk kasus yang
sama persis (deteksi gambar AI-generated di dataset CIFAKE): total
8 convolutional layer + 2 hidden (dense) layer. Catatan: paper tsb
tidak mempublikasikan tabel hyperparameter super detail (ukuran
kernel/filter tiap layer) di bagian yang bisa saya akses, jadi
implementasi berikut adalah desain umum & wajar yang mengikuti pola
tersebut (4 blok, tiap blok 2 conv layer), bukan replikasi 1:1 dari
kode aslinya. Jelaskan hal ini juga di laporan/skripsimu supaya jujur
soal mana yang "mengikuti ide paper" vs "hasil desain sendiri".

Total parameter model ini sekitar 300rb-600rb (tergantung filter),
jauh lebih kecil dari ResNet/VGG (puluhan juta parameter) -> cocok
disebut "algoritma ringan" di laporan kamu.
"""

from tensorflow.keras import layers, models

from . import config


def build_lightweight_cnn(input_shape=(32, 32, 3)) -> models.Model:
    inputs = layers.Input(shape=input_shape, name="image")

    x = inputs
    # 4 blok, masing-masing 2 conv layer -> total 8 conv layer
    filters_per_block = [32, 64, 128, 128]
    for i, f in enumerate(filters_per_block):
        x = layers.Conv2D(f, 3, padding="same", activation=None, name=f"block{i+1}_conv1")(x)
        x = layers.BatchNormalization(name=f"block{i+1}_bn1")(x)
        x = layers.Activation("relu", name=f"block{i+1}_act1")(x)

        x = layers.Conv2D(f, 3, padding="same", activation=None, name=f"block{i+1}_conv2")(x)
        x = layers.BatchNormalization(name=f"block{i+1}_bn2")(x)
        x = layers.Activation("relu", name=f"block{i+1}_act2")(x)

        # Jangan pooling lagi kalau feature map sudah kecil (image 32x32 -> setelah
        # 3x pooling jadi 4x4, blok ke-4 kita biarkan tanpa pooling tambahan)
        if i < 3:
            x = layers.MaxPooling2D(2, name=f"block{i+1}_pool")(x)
        x = layers.Dropout(0.25, name=f"block{i+1}_drop")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    # 2 hidden (dense) layer
    x = layers.Dense(128, activation="relu", name="hidden1")(x)
    x = layers.Dropout(0.5, name="hidden1_drop")(x)
    x = layers.Dense(64, activation="relu", name="hidden2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs, outputs, name="lightweight_cifake_cnn")
    return model


def compile_model(model: models.Model) -> models.Model:
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            "AUC",
            "Precision",
            "Recall",
        ],
    )
    return model


if __name__ == "__main__":
    m = build_lightweight_cnn(input_shape=(*config.IMG_SIZE, config.CHANNELS))
    m = compile_model(m)
    m.summary()
