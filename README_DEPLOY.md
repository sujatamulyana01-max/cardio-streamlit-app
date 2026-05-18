# Cardio Streamlit Deploy

Folder ini berisi file siap deploy untuk project Cardiovascular Data Modelling.

## Isi file
- `app.py`: aplikasi Streamlit
- `cardio_classifier.pkl`: model klasifikasi cardio
- `systolic_regressor.pkl`: model prediksi tekanan darah sistolik
- `model_metadata.json`: informasi fitur dan performa model
- `train_model.py`: script untuk melatih ulang model
- `requirements.txt`: daftar library

## Cara menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara deploy ke Streamlit Cloud
1. Buat repository GitHub baru.
2. Upload semua file dalam folder ini ke repository.
3. Buka Streamlit Community Cloud.
4. Klik Create app.
5. Pilih repository, branch, dan file utama `app.py`.
6. Klik Deploy.

## Performa model dari file yang dibuat
- Classification accuracy: 0.7376
- Classification F1 score: 0.7205
- Regression MAE: 7.20
- Regression RMSE: 10.59
- Regression R²: 0.6061

Catatan: aplikasi ini untuk edukasi/tugas, bukan diagnosis medis.
