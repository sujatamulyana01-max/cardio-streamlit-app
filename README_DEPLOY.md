# Cardio Prediction Streamlit App

Project ini adalah aplikasi sederhana berbasis Streamlit untuk memprediksi risiko penyakit kardiovaskular dan memperkirakan tekanan darah sistolik dari data kesehatan pengguna.

Aplikasi ini dibuat untuk kebutuhan deployment dan demo model machine learning. Jadi, output dari aplikasi lebih cocok digunakan sebagai simulasi atau bahan pembelajaran, bukan sebagai alat diagnosis medis.

## Fitur aplikasi

Beberapa fitur utama yang ada di aplikasi ini:

- Prediksi risiko cardio berdasarkan data input pengguna.
- Menampilkan probabilitas risiko dalam bentuk persentase.
- Menampilkan confidence level agar hasil prediksi lebih mudah dipahami.
- Prediksi tekanan darah sistolik menggunakan model regresi.
- Hasil regresi diberi label **Normal** atau **Tidak Normal**.
- Fitur comparison untuk membandingkan dua skenario input yang berbeda.

## Fitur comparison

Fitur comparison digunakan untuk membandingkan dua kondisi atau skenario. Misalnya, pengguna bisa melihat perbedaan hasil prediksi antara data A dan data B.

Bagian yang dibandingkan meliputi:

- BMI
- Pulse pressure
- Prediksi risiko cardio
- Risk probability
- Confidence level
- Prediksi tekanan darah sistolik
- Label hasil regresi

Fitur ini berguna supaya hasil prediksi tidak hanya muncul satu per satu, tetapi bisa dibandingkan secara langsung.

## Confidence level

Confidence level digunakan untuk menunjukkan seberapa yakin model terhadap hasil prediksi klasifikasi.

Pembagiannya dibuat seperti ini:

- **Tinggi**: confidence 75% ke atas
- **Sedang**: confidence 60% sampai di bawah 75%
- **Rendah**: confidence di bawah 60%

Confidence ini diambil dari probabilitas tertinggi yang diberikan oleh model klasifikasi.

## Label hasil regresi

Untuk hasil regresi, aplikasi tidak hanya menampilkan angka prediksi sistolik. Hasilnya juga diberi label agar lebih mudah dibaca.

Aturannya:

- **Normal** jika prediksi sistolik kurang dari 120 mmHg dan diastolik kurang dari 80 mmHg.
- **Tidak Normal** jika tidak memenuhi kondisi tersebut.

Label ini dibuat agar output regresi lebih jelas saat ditampilkan di aplikasi.

## File yang ada di project

| File | Keterangan |
|---|---|
| `app.py` | File utama aplikasi Streamlit |
| `cardio_classifier.pkl` | Model klasifikasi risiko cardio |
| `systolic_regressor.pkl` | Model regresi tekanan darah sistolik |
| `model_metadata.json` | Informasi fitur dan performa model |
| `health_data.csv` | Dataset yang digunakan |
| `train_model.py` | Script untuk training model |
| `requirements.txt` | Daftar library yang dibutuhkan |

## Cara menjalankan di laptop

Install dulu library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Lalu jalankan aplikasi:

```bash
streamlit run app.py
```

## Cara deploy ke Streamlit Cloud

1. Upload semua file project ke GitHub.
2. Masuk ke Streamlit Community Cloud.
3. Pilih repository GitHub yang sudah dibuat.
4. Pada bagian main file, pilih `app.py`.
5. Klik deploy.

## Hasil performa model

Performa model yang digunakan pada project ini:

| Model | Hasil |
|---|---:|
| Classification accuracy | 0.7376 |
| Classification F1 score | 0.7205 |
| Regression MAE | 7.20 mmHg |
| Regression RMSE | 10.59 mmHg |
| Regression R² | 0.6061 |

## Catatan

Aplikasi ini dibuat untuk keperluan pembelajaran, demo, dan deployment model. Hasil prediksi tidak boleh dijadikan dasar utama untuk keputusan medis.
