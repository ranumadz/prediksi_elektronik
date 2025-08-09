import streamlit as st
import pickle
import numpy as np

 
with open('model_rf_terbaik.pkl', 'rb') as f:
    model = pickle.load(f)


st.title("📦 Prediksi Lama Produksi Alat Elektronik")
st.markdown("Gunakan aplikasi ini untuk memprediksi waktu produksi berdasarkan beberapa faktor.")


jenis_produk = st.selectbox("Jenis Produk", ["Produk A", "Produk B", "Produk C"])
jenis_produk_encoded = {"Produk A": 0, "Produk B": 1, "Produk C": 2}[jenis_produk]

permintaan = st.number_input("Permintaan (unit)", min_value=0, step=1)
stok_bahan = st.number_input("Stok Bahan (unit)", min_value=0, step=1)
mesin_aktif = st.number_input("Jumlah Mesin Aktif", min_value=0, step=1)
tenaga_kerja = st.number_input("Jumlah Tenaga Kerja", min_value=0, step=1)


if st.button("🔍 Prediksi Lama Produksi"):
    input_data = np.array([[jenis_produk_encoded, permintaan, stok_bahan, mesin_aktif, tenaga_kerja]])
    prediksi = model.predict(input_data)
    st.success(f"⏱️ Perkiraan Lama Produksi: **{prediksi[0]:.2f} jam**")
