# Cardio Prediction Streamlit App

Repository ini berisi aplikasi Streamlit untuk prediksi penyakit kardiovaskular dan estimasi tekanan darah sistolik dari data kesehatan yang dimasukkan user.

## Fitur aplikasi

- Prediksi penyakit kardiovaskular.
- Probabilitas hasil prediksi dalam persentase.
- Confidence level agar output model lebih mudah dibaca.
- Prediksi tekanan darah sistolik dengan model regresi.
- Status tekanan darah: **Normal** atau **Tidak Normal**.
- Status tekanan darah disesuaikan dengan usia dan gender.
- Explainable AI untuk melihat feature yang paling berpengaruh.
- Comparison Feature untuk membandingkan feature importance dari model prediksi penyakit dan model prediksi sistolik.

## Comparison Feature

Comparison Feature dipakai untuk membandingkan feature yang paling berpengaruh menurut model.

Bagian ini menampilkan feature mana yang paling besar kontribusinya pada:

- model prediksi penyakit kardiovaskular;
- model prediksi tekanan darah sistolik.

Output ditampilkan dalam bentuk tabel dan grafik agar lebih mudah dibaca saat demo.

## Explainable AI

Explainable AI dibuat supaya hasil prediksi tidak hanya muncul sebagai angka. Aplikasi juga menampilkan feature teratas yang paling diperhatikan model.

Metode yang digunakan adalah **feature importance** dari model Gradient Boosting. Cara ini ringan untuk Streamlit dan tidak membutuhkan library tambahan yang berat.

Feature yang dapat muncul antara lain usia, berat badan, tekanan sistolik, tekanan diastolik, BMI, pulse pressure, kolesterol, glukosa, aktivitas fisik, dan status cardio.

## Confidence level

Confidence level menunjukkan tingkat keyakinan model terhadap hasil prediksi klasifikasi.

Pembagiannya:

- **Tinggi**: 75% ke atas
- **Sedang**: 60% sampai di bawah 75%
- **Rendah**: di bawah 60%

Confidence diambil dari probabilitas tertinggi yang diberikan oleh model klasifikasi.

## Status tekanan darah

Pada prediksi sistolik, aplikasi menampilkan angka prediksi dalam satuan mmHg dan statusnya.

Status **Normal** atau **Tidak Normal** dihitung dari:

- usia;
- gender;
- hasil prediksi sistolik;
- input tekanan diastolik.

Acuan status mengikuti rentang tekanan darah normal berdasarkan kelompok usia dan gender yang dimasukkan ke dalam aplikasi.

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

Install library:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

## Cara deploy ke Streamlit Cloud

1. Upload semua file project ke GitHub.
2. Buka Streamlit Community Cloud.
3. Pilih repository GitHub.
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
