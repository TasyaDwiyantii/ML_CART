import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page config
st.set_page_config(page_title="Prediksi Customer Churn", page_icon="📊", layout="wide")

# Load model
loaded_model = joblib.load('machine_learning/App/Cart-Customer_Churn.pkl')

# Fungsi utama aplikasi
def main():
    page = st.sidebar.radio("CART", ["Input", "Visualisasi"])

    if page == "Input":
        # Judul dan deskripsi aplikasi
        st.title("📊 Prediksi Customer Churn Menggunakan Metode CART")
        st.markdown("""        
        **Aplikasi ini memprediksi kemungkinan churn (kehilangan pelanggan) berdasarkan data pelanggan.**
        
        Silakan masukkan informasi pelanggan dengan benar dan klik tombol **'Prediksi'** untuk melihat hasil.
        """)
        
        # Divider untuk pemisah bagian
        st.markdown("---")

        # Bagian Input Data Pelanggan
        st.header("📝 Masukkan Data Pelanggan")

        # Kolom input untuk tata letak lebih baik
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("🗓️ Umur (Tahun)", min_value=0, value=None, placeholder="Masukkan umur pelanggan (tahun)")
            call_failure = st.number_input("📞 Call Failure", min_value=0, value=None, help="Masukkan jumlah kegagalan panggilan", placeholder="Masukkan jumlah kegagalan panggilan")
            complains = st.selectbox("📞 Complaints", options=[0, 1], format_func=lambda x: "0 (tidak ada keluhan)" if x == 0 else "1 (ada keluhan)", index=0, placeholder="Pilih 0 atau 1")
            subscription_length = st.number_input("📅 Subscription Length (bulan)", min_value=0, value=None, placeholder="Masukkan durasi langganan (bulan)")
            charge_amount = st.number_input("💰 Charge Amount (0-9)", min_value=0, value=None, placeholder="Masukkan jumlah tagihan (0 : biaya terendah, 9 : biaya tertinggi)")

        with col2:
            seconds_of_use = st.number_input("⏱️ Seconds of Use", min_value=0, value=None, placeholder="Masukkan waktu penggunaan (detik)")
            frequency_of_use = st.number_input("📞 Frequency of Use", min_value=0, value=None, placeholder="Masukkan frekuensi penggunaan")
            frequency_of_sms = st.number_input("📱 Frequency of SMS", min_value=0, value=None, placeholder="Masukkan frekuensi SMS")
            distinct_called_numbers = st.number_input("🔢 Distinct Called Numbers", min_value=0, value=None, placeholder="Masukkan jumlah nomor yang dihubungi")
            age_group = st.selectbox("👥 Age Group", options=[1, 2, 3, 4, 5], index=0, help="Pilih kelompok usia (1 paling muda, 5 paling tua).", placeholder="Pilih kelompok usia (1 paling muda, 5 paling tua)")

        with col3:
            tariff_plan = st.selectbox("💸 Tariff Plan",  options=[1, 2], format_func=lambda x: "1 (bayar sesuai yang dipakai)" if x == 1 else "2 (kontraktual)", index=0, placeholder="Pilih paket tarif")
            status = st.selectbox("🔴 Status", options=[1, 2], format_func=lambda x: "1 (Aktif)" if x == 1 else "2 (Tidak Aktif)", index=0, placeholder="Pilih status pelanggan")
            customer_value = st.number_input("💵 Customer Value", min_value=0.0, value=None, placeholder="Masukkan nilai pelanggan")


        # Menyusun data input untuk prediksi
        # Tidak perlu konversi numerik untuk age_group, tariff_plan, status karena sudah berupa numerik
        # Semua variabel sudah dikonversi sesuai input

        # Tombol Prediksi
        if st.button("🔍 Prediksi", use_container_width=True):
            # Prediksi menggunakan model yang dimuat
            prediction = loaded_model.predict([[call_failure, complains, subscription_length, charge_amount,
                                                seconds_of_use, frequency_of_use, frequency_of_sms,
                                                distinct_called_numbers, age_group, tariff_plan,
                                                status, age, customer_value]])

            # Menampilkan hasil prediksi dengan styling
            if prediction[0] == 1:
                label = "⚠️ Pelanggan Cenderung Berhenti"
                color = "red"
            else:
                label = "✅ Pelanggan Tidak Berhenti"
                color = "green"

            st.markdown(f"<h2 style='color: {color}; text-align: center;'>{label}</h2>", unsafe_allow_html=True)

        # Footer
        st.markdown("""        
        <br><br>
        <hr>
        <div style="text-align: center;">
            <p>© 2024 Aplikasi Prediksi Customer Churn | Dibuat oleh Tasya Dwiyanti</p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "Visualisasi":
        # Visualisasi pohon keputusan (CART)
        import matplotlib.pyplot as plt
        from sklearn import tree

        feature_names = [
            "Call Failure", "Complains", "Subscription Length", "Charge Amount", "Seconds of Use", "Frequency of use", 
            "Frequency of SMS", "Distinct Called Numbers", "Age Group", "Tariff Plan", "Status", "Age", "Customer Value"
        ]
        
        # Menampilkan visualisasi pohon keputusan
        fig = plt.figure(figsize=(20, 20))  # Sesuaikan ukuran gambar
        _ = tree.plot_tree(loaded_model,
                           feature_names=feature_names,
                           class_names=["0", "1"],  # "0" = Tidak Churn, "1" = Churn
                           fontsize=5,
                           filled=True)

        # Menampilkan visualisasi
        st.pyplot(fig)

# Jalankan aplikasi
if __name__ == "__main__":
    main()
