# OTENTIK — Pemindai Gambar & Wajah Asli vs Buatan AI

Tugas akhir: mendeteksi apakah sebuah gambar/wajah itu asli atau buatan
AI, memakai CNN ringan (608 ribu parameter) yang dilatih sendiri dari
nol lalu di-transfer-learning ke domain wajah, di-deploy sebagai web app
mandiri (TensorFlow.js) yang jalan sepenuhnya di perangkat pengguna.

**📄 Baca dulu:** [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md) —
ringkasan lengkap jurnal yang dipakai, spesifikasi model, proses
training, konversi ke TensorFlow.js, sampai temuan & keterbatasannya.
Cocok jadi bahan bab Metodologi/Hasil/Pembahasan skripsi.

**🚀 Mau upload ke GitHub / lanjut training di Colab?** Lihat
[`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md).

## Coba langsung

Buka [`app/otentik_face.html`](app/otentik_face.html) di browser
(termasuk di HP) — tidak perlu install apapun, model sudah ter-embed di
dalam filenya.

## Struktur proyek

```
├── docs/               <- dokumentasi lengkap (baca ini duluan)
├── notebooks/          <- notebook Google Colab (training di GPU gratis)
├── src/                <- kode training (config, dataset, model, train, evaluate, export)
├── app/                <- web app siap pakai + demo Streamlit
├── data/               <- taruh dataset di sini (lihat data/README_DATA.md)
├── models/             <- model hasil training tersimpan di sini
└── requirements.txt
```

## Quick start (training ulang / lanjutan)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m src.train
python -m src.evaluate
python -m src.export_tflite
```

Atau langsung pakai notebook di `notebooks/` lewat Google Colab (lebih
cepat, ada GPU gratis) — lihat instruksi di dalam masing-masing notebook.

## Referensi utama

- Bird, J.J. & Lotfi, A. (2024). *CIFAKE: Image Classification and
  Explainable Identification of AI-Generated Synthetic Images*. IEEE Access.
- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator
  Architecture for Generative Adversarial Networks*. CVPR.
- Dataset: [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) (xhlulu, Kaggle)

Daftar referensi lengkap ada di [`docs/PROJECT_SUMMARY.md`](docs/PROJECT_SUMMARY.md#2-jurnal--sumber-ilmiah-yang-dipakai).
