# Cardio Prediction Streamlit App

Project ini berisi aplikasi Streamlit untuk memprediksi penyakit kardiovaskular dan memperkirakan tekanan darah sistolik dari input data kesehatan.

Aplikasi ini dipakai untuk demo deployment model machine learning. Hasilnya tetap bersifat edukasi, jadi tidak boleh dijadikan pengganti pemeriksaan dokter.

## Fitur aplikasi

- Prediksi penyakit kardiovaskular dari data pasien.
- Probabilitas penyakit dalam bentuk persentase.
- Confidence level agar hasil model lebih mudah dibaca.
- Prediksi tekanan darah sistolik menggunakan model regresi.
- Status hasil tekanan darah: **Normal** atau **Tidak Normal**.
- Explainable AI untuk melihat feature yang paling berpengaruh.
- Comparison Feature untuk membandingkan feature importance dari model prediksi penyakit dan model prediksi sistolik.

## Comparison Feature

Pada aplikasi ini, Comparison Feature bukan untuk membandingkan dua pasien. Bagian ini dipakai untuk membandingkan feature yang paling berpengaruh menurut model.

Jadi user bisa melihat feature mana yang paling besar kontribusinya pada:

- model prediksi penyakit kardiovaskular;
- model prediksi tekanan darah sistolik.

Output-nya ditampilkan dalam bentuk tabel dan grafik supaya lebih mudah dibaca saat demo.

## Explainable AI

Explainable AI dibuat supaya hasil prediksi tidak hanya muncul sebagai angka. Aplikasi juga menampilkan feature teratas yang paling diperhatikan model.

Metode yang dipakai adalah **feature importance** dari model. Pendekatan ini ringan untuk Streamlit, tidak perlu library tambahan yang berat, dan cukup jelas untuk kebutuhan demo deployment.

Contoh feature yang bisa muncul antara lain usia, berat badan, tekanan sistolik, tekanan diastolik, BMI, pulse pressure, kolesterol, glukosa, dan aktivitas fisik.

Feature importance hanya menjelaskan pola model. Artinya, feature yang muncul paling tinggi bukan berarti pasti menjadi penyebab medis secara langsung.

## Confidence level

Confidence level menunjukkan seberapa yakin model terhadap hasil prediksi klasifikasi.

Pembagiannya:

- **Tinggi**: 75% ke atas
- **Sedang**: 60% sampai di bawah 75%
- **Rendah**: di bawah 60%

Confidence diambil dari probabilitas tertinggi yang diberikan oleh model klasifikasi.

## Status hasil regresi

Untuk prediksi tekanan darah sistolik, aplikasi menampilkan angka prediksi dalam satuan mmHg dan statusnya.

Aturannya:

- **Normal** jika prediksi sistolik kurang dari 120 mmHg dan diastolik kurang dari 80 mmHg.
- **Tidak Normal** jika tidak memenuhi kondisi tersebut.

Status ini dibuat supaya output regresi lebih mudah dipahami saat aplikasi dijalankan.

## File project

| File | Keterangan |
|---|---|
| `app.py` | File utama aplikasi Streamlit |
| `cardio_classifier.pkl` | Model klasifikasi penyakit kardiovaskular |
| `systolic_regressor.pkl` | Model regresi tekanan darah sistolik |
| `model_metadata.json` | Informasi fitur dan performa model |
| `health_data.csv` | Dataset yang digunakan |
| `train_model.py` | Script training model |
| `requirements.txt` | Library yang dibutuhkan |

## Cara menjalankan di laptop

Install library dulu:

```bash
pip install -r requirements.txt
```

Jalankan aplikasinya:

```bash
streamlit run app.py
```

## Cara deploy ke Streamlit Cloud

1. Upload semua file project ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub yang sudah dibuat.
4. Isi main file dengan `app.py`.
5. Klik deploy.

## Performa model

| Model | Hasil |
|---|---:|
| Classification accuracy | 0.7376 |
| Classification F1 score | 0.7205 |
| Regression MAE | 7.20 mmHg |
| Regression RMSE | 10.59 mmHg |
| Regression R² | 0.6061 |

## Catatan

