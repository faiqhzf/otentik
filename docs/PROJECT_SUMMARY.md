# OTENTIK — Ringkasan Proyek Lengkap

Dokumen ini merangkum seluruh perjalanan proyek tugas akhir: dari ide awal,
riset jurnal, training model, sampai deployment ke web/mobile — termasuk
apa yang berhasil, apa yang gagal, dan kenapa. Ditulis supaya bisa langsung
jadi bahan bab Metodologi/Hasil/Pembahasan di skripsi.

---

## 1. Ide & Rumusan Masalah

Konten buatan AI (gambar, wajah) makin sulit dibedakan dari yang asli di
media sosial. Tugas akhir ini membangun **detektor ringan** yang bisa
dilatih sendiri, berjalan di perangkat (bukan di server), dan dipasang
sebagai aplikasi web/mobile sederhana.

Proyek ini berkembang melalui **3 fase**, masing-masing dengan keputusan
teknis dan temuan yang perlu didokumentasikan:

| Fase | Fokus | Hasil |
|---|---|---|
| 1 | Deteksi gambar AI umum (dataset CIFAKE) | Akurasi 96,53%, tapi gagal generalisasi ke foto wajah manusia |
| 2 | Transfer learning ke domain wajah (140k Real/Fake Faces) | Akurasi 97,10%, tapi masih ada isu preprocessing |
| 3 | Deployment web (TensorFlow.js) + deteksi wajah otomatis | Berhasil sebagian; ditemukan batas nyata pada model ringan |

---

## 2. Jurnal & Sumber Ilmiah yang Dipakai

### Dataset & baseline deteksi gambar AI
- Bird, J.J. & Lotfi, A. (2024). *CIFAKE: Image Classification and
  Explainable Identification of AI-Generated Synthetic Images*. IEEE
  Access. arXiv:2303.14126. — sumber dataset CIFAKE, baseline CNN 92,98%.
- Lokner Ladjevic, A., Kramberger, T., Kramberger, R., & Vlahek, D.
  (2024). *Detection of AI-Generated Synthetic Images with a Lightweight
  CNN*. AI (MDPI), 5(3), 1575–1593. — inspirasi arsitektur (8 conv layer +
  2 dense layer), akurasi 97,32% di CIFAKE.

### Dataset & alignment wajah
- Dataset: **140k Real and Fake Faces** (xhlulu, Kaggle) — 70.000 wajah
  asli (FFHQ) + 70.000 wajah StyleGAN.
- Karras, T., Laine, S., & Aila, T. (2019). *A Style-Based Generator
  Architecture for Generative Adversarial Networks*. CVPR, 4396–4405. —
  sumber dataset FFHQ sekaligus **algoritma resmi alignment wajah**
  (`face_alignment.py`, repo NVlabs/ffhq-dataset) yang direplikasi persis
  di `app/otentik_face.html`.

### Deployment & preprocessing wajah
- Deteksi wajah di browser: **BlazeFace** (`@tensorflow-models/blazeface`,
  Google/TensorFlow.js).
- Praktik preprocessing (deteksi→crop→align sebelum klasifikasi) mengacu
  ke metodologi umum di literatur deepfake detection (mis. DeepfakeBench,
  NeurIPS 2023 benchmark) yang menyebutkan deteksi & alignment wajah
  sebagai langkah wajib sebelum klasifikasi.

### Generalisasi & limitasi model deteksi AI
- Chandra, S. dkk. (2025). *Deepfake-Eval-2024*. arXiv:2503.02857. —
  menemukan akurasi (AUC) detektor deepfake turun ~45-50% saat diuji ke
  konten dunia-nyata dibanding dataset akademis asalnya.
- Guo, C. dkk. — soal jaringan saraf yang cenderung overconfident
  (yakin tinggi tapi salah) pada data di luar distribusi latihnya —
  fenomena yang berulang kali teramati di pengujian proyek ini.

### Arsitektur referensi (dibahas, tidak dipakai langsung)
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
  *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR,
  4510–4520. — referensi arsitektur mobile-friendly untuk perbandingan.

---

## 3. Spesifikasi Model

- **Arsitektur**: CNN custom, 8 convolutional layer (4 blok × 2 conv,
  dengan BatchNorm + ReLU + MaxPooling + Dropout tiap blok) +
  GlobalAveragePooling2D + 2 dense layer (128 → 64) + output sigmoid.
- **Total parameter**: 608.417 (~2,3 MB dalam float32) — masuk kategori
  "ringan" dibanding ResNet/VGG (puluhan juta parameter).
- **Sifat penting**: karena pakai `GlobalAveragePooling2D` sebelum dense
  layer, bobot model **tidak bergantung pada resolusi spasial input** —
  ini yang memungkinkan transfer learning dari 32×32 (CIFAKE) ke 128×128
  (wajah) tanpa mismatch shape (sudah diverifikasi langsung).

| | Model Fase 1 (CIFAKE) | Model Fase 2 (Wajah) |
|---|---|---|
| Input | 32×32×3 | 128×128×3 |
| Parameter | 608.417 | 608.417 (sama, arsitektur identik) |
| Inisialisasi bobot | Acak | Warm-start dari model Fase 1 |
| Akurasi uji | 96,53% | 97,10% |
| ROC-AUC | 0,9945 | 0,9976 |
| Ukuran .tflite (quantized) | 617 KB | 631 KB |

---

## 4. Proses Training

**Fase 1 (CIFAKE)**: 30 epoch, Adam optimizer, EarlyStopping +
ModelCheckpoint(save_best_only) + ReduceLROnPlateau, augmentasi ringan
(flip/rotate/zoom). Dilatih di Google Colab (GPU T4).

**Fase 2 (Wajah, transfer learning)**: bobot model Fase 1 dimuat lebih
dulu (`model.load_weights()`) sebelum training dimulai, learning rate
lebih kecil (3e-4 vs 1e-3) karena sudah warm-start, 19 epoch sampai
early stopping terpicu. Checkpoint terbaik diambil dari **epoch 15**
(val_loss terendah ≈0,075) — bukan epoch terakhir, meski grafik training
terlihat sedikit naik-turun setelahnya (ini penting: `ModelCheckpoint` +
`EarlyStopping(restore_best_weights=True)` sudah menangani ini otomatis).

Confusion matrix Fase 2 menunjukkan sedikit ketimpangan: recall wajah
"fake" 98,98% vs recall wajah "real" 95,22% — model sedikit lebih sering
salah menuduh wajah asli sebagai AI dibanding sebaliknya.

---

## 5. Proses Konversi ke TensorFlow.js (untuk web/mobile)

Model `.keras` (format Keras 3) dikonversi ke TensorFlow.js lewat jalur:
`.keras` → `.h5` (format lama) → `tensorflowjs_converter` (dengan
`--quantize_float16`) → `model.json` + file bobot biner.

**Masalah yang ditemukan & diperbaiki** (Keras 3 mengubah beberapa format
JSON dibanding Keras 2 yang diharapkan TensorFlow.js):
- `batch_shape` (Keras 3) harus diubah jadi `batch_input_shape` (format lama).
- `dtype` berupa objek `DTypePolicy` harus diubah jadi string biasa (`"float32"`).
- `inbound_nodes` (struktur graph antar-layer) berubah total formatnya di
  Keras 3 (dari list-of-list ke dict dengan `args`/`kwargs`) — perlu
  dikonversi manual.
- `input_layers`/`output_layers` di Keras 3 berupa triple langsung,
  sedangkan format lama mengharapkan list-of-triple.

Semua perbaikan ini ada di script `patch_keras3_to_tfjs.py` (bagian dari
proses build, tidak disertakan di repo karena sudah "ditempel" ke proses
konversi — lihat bagian rekomendasi soal ini di bawah).

**Verifikasi**: setiap kali model dikonversi, prediksi dari model asli
(Python/Keras) dan hasil konversi (TensorFlow.js) dibandingkan dengan
input yang identik — selisihnya konsisten di orde 1e-7 sampai 1e-9
(sekadar pembulatan floating point, bukan perbedaan berarti).

---

## 6. Web App (OTENTIK) — Evolusi & Temuan Penting

`app/otentik_face.html` adalah satu file HTML mandiri (model ter-embed
sebagai base64, ~1,6 MB) yang bisa dibuka langsung di browser HP tanpa
instalasi apapun. Perjalanannya:

1. **v1**: resize foto utuh apa adanya ke 128×128 → banyak salah deteksi
   pada selfie/foto grup (wajah cuma sebagian kecil dari frame).
2. **v2**: ditambah **BlazeFace** (deteksi wajah) + crop ke wajah — tapi
   sempat ada bug: elemen `<img>` yang di-CSS (`object-fit:cover`) bikin
   BlazeFace cuma "melihat" potongan pojok foto, bukan foto penuh. Sudah
   diperbaiki dengan menggambar ulang ke kanvas resolusi native
   (`naturalWidth`/`naturalHeight`) sebelum dikirim ke BlazeFace.
3. **v3**: alignment wajah diperbaiki pakai **rumus resmi FFHQ**
   (bukan perkiraan sendiri) — level kemiringan kepala & skala berdasar
   jarak mata-mulut, meniru persis cara dataset training disiapkan.
4. **Temuan penting (v3)**: untuk sebagian foto (terutama foto studio
   dengan background rata), **foto utuh justru lebih akurat** daripada
   wajah yang di-crop — karena ciri AI-nya ada di background/komposisi,
   bukan di tekstur wajah. Dicoba digabung jadi satu skor (ambil yang
   paling curiga, atau dirata-rata) — **tidak ada rumus gabungan yang
   konsisten benar** di semua kasus uji.
5. **Keputusan akhir**: app menampilkan **dua skor terpisah** (foto utuh
   vs wajah ter-crop) alih-alih memaksa satu angka, dan memberi peringatan
   eksplisit kalau keduanya tidak sepakat.

Ini adalah temuan yang jujur dan bernilai ilmiah: **model ringan
tunggal punya batas nyata** dalam memutuskan "sudut pandang" mana yang
harus dipercaya, dan preprocessing yang "lebih canggih" tidak otomatis
berarti "lebih akurat" di semua kasus.

---

## 7. Keterbatasan yang Terbukti Lewat Pengujian

- Model gagal pada foto ilustrasi/lukisan (domain gaya berbeda dari data
  latih yang seluruhnya foto-realistis).
- Model kadang gagal pada foto studio dengan background rata/AI headshot
  tertentu — kemungkinan besar karena generator yang dipakai berbeda dari
  StyleGAN (satu-satunya generator di data training), konsisten dengan
  temuan Deepfake-Eval-2024 soal buruknya generalisasi lintas-generator.
- Tidak ada rumus gabungan foto-utuh vs wajah-crop yang selalu benar.
- BlazeFace dioptimalkan untuk wajah frontal jarak dekat gaya kamera HP —
  performa bisa menurun di foto grup dengan banyak wajah kecil.

---

## 8. Saran Pengembangan Lebih Lanjut

1. **Dataset lebih beragam generator**: tambahkan contoh wajah AI dari
   Midjourney/DALL-E/diffusion model terbaru (tidak cuma StyleGAN), untuk
   mengatasi masalah generalisasi cross-generator yang ditemukan.
2. **Model terpisah untuk "konteks luar wajah"**: alih-alih 1 model
   dipakai untuk 2 sudut pandang (foto utuh & wajah), pertimbangkan
   melatih 2 model terpisah yang masing-masing dioptimalkan untuk
   perannya, lalu digabung dengan strategi ensemble yang dilatih (bukan
   rumus manual seperti min/rata-rata yang terbukti tidak konsisten).
3. **Investigasi kenapa background jadi sinyal**: analisis lebih lanjut
   (mis. Grad-CAM/saliency map) untuk melihat bagian mana dari gambar
   yang paling memengaruhi keputusan model — akan memperkuat pembahasan
   soal temuan "sinyal di luar wajah" di bab hasil.
4. **Aplikasi mobile native**: file `.tflite` di `models/` (hasil
   `export_tflite.py`) sudah siap dipakai di Flutter (`tflite_flutter`)
   atau Android native, kalau mau lanjut dari prototipe web ke aplikasi
   terpasang beneran.
5. **Uji coba threshold BlazeFace**: `scoreThreshold` saat ini diturunkan
   ke 0,5 supaya lebih toleran mendeteksi wajah di foto grup — perlu
   diuji lebih sistematis apakah ini menambah salah-deteksi di kasus lain.

---

## 9. Fase 4: Sistem Pakar untuk Menggabungkan Dua Sinyal

Menindaklanjuti temuan di bagian 6 (tidak ada rumus matematis tetap yang
konsisten menggabungkan sinyal foto-utuh dan wajah-ter-crop), `app/otentik_face.html`
sekarang memakai **sistem pakar** untuk penggabungan ini, bukan rumus tetap.

### Basis pengetahuan (rule base)
Fakta yang diekstrak otomatis dari tiap gambar: rasio luas wajah terhadap
frame, skor keyakinan deteksi BlazeFace, jumlah wajah terdeteksi, dan
**keseragaman warna di tepi gambar** (proksi untuk "background rata",
dihitung dari standar deviasi piksel di 4 sisi tepi foto). Lima aturan
IF-THEN dievaluasi (forward chaining) untuk menimbang ulang kedua sinyal
sebelum digabung:

| Aturan | Kondisi | Efek |
|---|---|---|
| R1 | Tidak ada wajah terdeteksi | Verdict pakai sinyal foto-utuh saja |
| R2 | Background rata (uniformity > 0,7) | Bobot sinyal foto-utuh dinaikkan 1,6× |
| R3 | Wajah < 15% luas frame | Bobot sinyal wajah diturunkan 0,5× |
| R4 | Skor keyakinan BlazeFace < 0,8 | Bobot sinyal wajah diturunkan 0,6× |
| R5 | > 3 wajah terdeteksi (foto grup) | Catatan transparansi (bobot tak berubah) |

### Mesin inferensi: Certainty Factor (gaya MYCIN)
Setiap probabilitas CNN (0–1) diubah jadi *Certainty Factor* (CF, −1
sampai +1) lewat `CF = 2×prob − 1`, ditimbang sesuai aturan yang aktif,
lalu digabung pakai rumus kombinasi CF resmi (Shortliffe & Buchanan,
1975, dipakai pertama kali di sistem pakar medis MYCIN):

```
CF1,CF2 ≥ 0  :  CF = CF1 + CF2×(1−CF1)
CF1,CF2 < 0  :  CF = CF1 + CF2×(1+CF1)
berlawanan   :  CF = (CF1+CF2) / (1−min(|CF1|,|CF2|))
```

### Fasilitas penjelasan (explanation facility)
Setiap keputusan menyertakan jejak lengkap aturan mana yang aktif dan
kenapa (bisa dilihat lewat tombol "Lihat penalaran sistem pakar" di
aplikasi) — ciri khas sistem pakar dibanding CNN murni yang black-box.

### Kenapa ini dipilih dibanding rumus tetap
Rumus tetap (`min()`, rata-rata) yang dicoba sebelumnya masing-masing
punya kasus gagal yang berbeda (lihat bagian 6 & 7) karena bobot yang
"benar" untuk tiap sinyal **bergantung konteks gambar** — persis jenis
masalah yang cocok diselesaikan sistem pakar berbasis aturan, bukan satu
formula statis. Semua rumus CF & rule engine sudah diverifikasi lewat
pengujian unit (kombinasi komutatif, penguatan bukti searah, tiap aturan
aktif tepat pada kondisinya) sebelum dipasang ke aplikasi.

### Saran pengembangan lanjutan untuk sistem pakar ini
- **Latih bobot aturan**, bukan nilai tetap (1,6× / 0,5× / 0,6×) — bisa
  dioptimalkan dari data uji berlabel supaya tidak lagi berbasis intuisi.
- **Tambah aturan baru** seiring ditemukannya pola kegagalan baru (mis.
  aturan khusus foto dengan pencahayaan ekstrem).
- **Fuzzy logic** sebagai alternatif Certainty Factor — kalau mau
  transisi antar-aturan lebih halus (bukan ambang batas keras seperti
  `uniformity > 0.7`).

