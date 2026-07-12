# Panduan Upload ke GitHub & Pakai dari Google Colab

## 1. Kode apa saja yang perlu di-upload

Struktur folder ini (`otentik-repo/`) sudah dirapikan supaya siap upload
langsung. Ringkasan isinya:

```
otentik-repo/
├── README.md                          <- halaman utama repo
├── docs/
│   └── PROJECT_SUMMARY.md             <- ringkasan lengkap (jurnal, spek, training, dst)
├── notebooks/
│   ├── 01_cifake_training_colab.ipynb <- notebook fase 1 (CIFAKE)
│   └── 02_face_finetune_training_colab.ipynb <- notebook fase 2 (wajah, transfer learning)
├── src/                                <- SEMUA kode training (fase terbaru/wajah)
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── export_tflite.py
│   └── predict.py
├── app/
│   ├── otentik_face.html              <- web app final (dual-signal + BlazeFace)
│   ├── otentik_cifake_v1.html         <- web app versi awal (arsip/pembanding)
│   └── demo_streamlit.py
├── data/
│   └── README_DATA.md                 <- instruksi download dataset (bukan dataset-nya sendiri)
├── requirements.txt
└── .gitignore
```

**Yang SENGAJA tidak ikut ter-upload** (sudah diatur di `.gitignore`):
- Isi folder `data/train`, `data/valid`, `data/test` (dataset asli —
  besar, dan bisa didownload ulang kapan saja dari Kaggle lewat notebook).
- File model hasil training (`.keras`, `.tflite`, `.h5`) — lihat catatan
  di bagian 4 soal ke mana file-file ini sebaiknya disimpan.
- Grafik/laporan hasil training (`reports/*.png`, `*.csv`, `*.txt`) —
  akan ter-generate ulang tiap kali training dijalankan.

## 2. Langkah upload (dari VS Code / terminal Linux)

```bash
# 1. Masuk ke folder proyek
cd otentik-repo

# 2. Inisialisasi git (kalau belum pernah)
git init
git add .
git commit -m "Initial commit: OTENTIK - deteksi gambar/wajah asli vs AI"

# 3. Buat repo kosong di GitHub
#    Buka https://github.com/new, isi nama repo (mis. "otentik-ai-detector"),
#    JANGAN centang "Add README" (biar tidak konflik dengan punya kita),
#    lalu klik Create repository.

# 4. Hubungkan repo lokal ke GitHub (ganti USERNAME & NAMA-REPO)
git remote add origin https://github.com/USERNAME/NAMA-REPO.git
git branch -M main
git push -u origin main
```

Kalau diminta login, GitHub sekarang pakai **Personal Access Token**
(bukan password biasa) — buat di
https://github.com/settings/tokens → **Generate new token (classic)** →
centang scope `repo` → pakai token itu sebagai password saat `git push`.

## 3. Update berikutnya (setelah edit kode)

```bash
git add .
git commit -m "deskripsi perubahan, mis: perbaiki alignment wajah"
git push
```

## 4. Soal file model (.keras / .tflite) yang besar

File model **sengaja tidak di-commit langsung** ke git (praktik yang
kurang disarankan untuk file binary yang sering berubah tiap training).
Dua opsi:

- **Opsi A (disarankan untuk skripsi)**: upload manual lewat
  **GitHub Releases** (tab "Releases" di halaman repo → "Create a new
  release" → drag & drop file `.keras`/`.tflite`). Ini juga otomatis
  memberi link download permanen yang bisa dicantumkan di skripsi.
- **Opsi B**: hapus baris terkait di `.gitignore` kalau memang mau
  commit langsung (aman untuk ukuran file kita yang cuma beberapa MB,
  jauh di bawah limit GitHub 100MB) — tapi riwayat repo akan cepat
  membesar tiap kali retraining.

## 5. Pakai dari Google Colab (jauh lebih simpel dari notebook lama)

Setelah kode ada di GitHub, notebook Colab **tidak perlu lagi** puluhan
cell `%%writefile` seperti sebelumnya — cukup clone repo-nya:

```python
!git clone https://github.com/USERNAME/NAMA-REPO.git
%cd NAMA-REPO
!pip install -q kaggle scikit-learn pillow matplotlib
```

Lalu proses training/evaluasi/export jalan seperti biasa:
```python
!python -m src.train
!python -m src.evaluate
!python -m src.export_tflite
```

Kalau di Colab kamu bikin perubahan kode dan mau disimpan balik ke
GitHub, dari dalam Colab:
```python
!git config --global user.email "emailmu@example.com"
!git config --global user.name "Nama Kamu"
!git add -A
!git commit -m "update dari Colab"
!git push https://<TOKEN>@github.com/USERNAME/NAMA-REPO.git main
```
(Ganti `<TOKEN>` dengan Personal Access Token yang sama seperti di
langkah 2 — jangan taruh token ini di kode yang di-share ke orang lain.)

## 6. Kenapa ini bikin pengembangan lebih gampang

- Edit kode di satu tempat (lokal/VS Code), tinggal `git push`, lalu
  `git clone`/`git pull` di Colab — tidak perlu re-generate notebook
  raksasa tiap kali ada perubahan kecil.
- Riwayat perubahan (`git log`) otomatis jadi dokumentasi perkembangan
  proyek — berguna untuk lampiran skripsi ("log pengembangan").
- Orang lain (dosen penguji, misalnya) bisa langsung `git clone` dan
  coba jalankan sendiri tanpa perlu file-file terpisah dikirim manual.
