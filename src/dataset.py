"""
dataset.py
Memuat gambar dari folder train/, valid/, test/ menjadi tf.data.Dataset.
Beda dari versi CIFAKE: dataset wajah ini sudah datang dengan split
train/valid/test dari pembuatnya, jadi tidak perlu potong validation_split
sendiri dari train/.
"""

import tensorflow as tf
from tensorflow.keras import layers

from . import config


def _prepare(ds: tf.data.Dataset, augment: bool, normalize: layers.Layer) -> tf.data.Dataset:
    ds = ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.05),
            layers.RandomContrast(0.05),
        ])
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def load_datasets():
    """
    Return: (train_ds, val_ds, test_ds)
    Label mengikuti urutan alfabetis nama folder -> fake=0, real=1
    (sudah didefinisikan juga di config.CLASS_NAMES).
    """
    normalize = layers.Rescaling(1.0 / 255)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.VALID_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="binary",
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
    )

    train_ds = _prepare(train_ds, augment=True, normalize=normalize)
    val_ds = _prepare(val_ds, augment=False, normalize=normalize)
    test_ds = _prepare(test_ds, augment=False, normalize=normalize)

    return train_ds, val_ds, test_ds
