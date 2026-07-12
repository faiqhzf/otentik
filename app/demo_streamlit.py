"""
demo_streamlit.py
Demo web sederhana: upload gambar -> model bilang REAL atau FAKE.

Jalankan dari root proyek:
    streamlit run app/demo_streamlit.py

Setelah jalan, Streamlit akan kasih 2 alamat:
  - Local URL   -> buka di laptop yang sama
  - Network URL -> buka dari HP asal masih di WiFi yang sama
                   (ini cara paling gampang untuk "demo di HP" tanpa
                   perlu bikin aplikasi Android/iOS native dulu)
"""

import os
import sys

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config  # noqa: E402


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(config.BEST_CHECKPOINT_PATH)


def predict(model, image: Image.Image) -> float:
    img = image.convert("RGB").resize(config.IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    batch = np.expand_dims(arr, axis=0)
    prob_real = float(model.predict(batch, verbose=0)[0][0])
    return prob_real


def main():
    st.set_page_config(page_title="Deteksi Gambar AI vs Asli", page_icon="🖼️")
    st.title("🖼️ Deteksi Gambar: Asli vs Buatan AI")
    st.caption(
        "Model CNN ringan (transfer learning dari CIFAKE), dilatih ulang di "
        "dataset 140k Real and Fake Faces — untuk tujuan edukasi/tugas akhir."
    )

    if not os.path.exists(config.BEST_CHECKPOINT_PATH):
        st.error(
            f"Model belum ditemukan di `{config.BEST_CHECKPOINT_PATH}`.\n\n"
            "Jalankan dulu `python -m src.train` sampai selesai."
        )
        return

    model = load_model()

    uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with st.spinner("Menganalisis ..."):
            prob_real = predict(model, image)

        label = "REAL (foto asli)" if prob_real >= 0.5 else "FAKE (dibuat AI)"
        confidence = prob_real if prob_real >= 0.5 else 1 - prob_real

        if prob_real >= 0.5:
            st.success(f"**{label}**  —  confidence {confidence*100:.1f}%")
        else:
            st.warning(f"**{label}**  —  confidence {confidence*100:.1f}%")

        st.caption(
            "Catatan: model dilatih di gambar wajah yang di-crop rapi (gaya FFHQ), "
            "jadi paling akurat untuk foto wajah frontal & jelas. Untuk foto grup, "
            "sudut ekstrem, atau generator AI yang tidak terwakili di data latih, "
            "akurasi bisa turun — ini termasuk keterbatasan yang wajar untuk "
            "dibahas di bab kesimpulan skripsi."
        )


if __name__ == "__main__":
    main()
