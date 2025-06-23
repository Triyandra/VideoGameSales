import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# --- Load dan pra-pemrosesan ---
@st.cache_data
def load_and_process_data():
    # Load dataset
    df = pd.read_csv('vgsales.csv')

    # Bersihkan kolom
    df_clean = df.drop(['Rank', 'Publisher'], axis=1)
    df_clean['Year'] = df_clean['Year'].fillna(df_clean['Year'].median())

    # Simpan nama untuk referensi
    df_names = df_clean['Name']

    # Drop Name untuk fitur
    df_clean = df_clean.drop(['Name'], axis=1)

    # One-hot encoding
    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'])

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)

    return df, df_names, X_scaled

# --- Latih model ---
@st.cache_resource
def train_model(X_scaled):
    model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    model.fit(X_scaled)
    return model

# --- Main app ---
st.title("🎮 Sistem Rekomendasi Video Game Berdasarkan Data Penjualan")
st.write("Pilih game untuk mendapatkan rekomendasi game serupa berdasarkan fitur penjualan.")

# Load data dan model
df, df_names, X_scaled = load_and_process_data()
nn_model = train_model(X_scaled)

# UI: pilih game
game_list = df_names.tolist()
selected_game = st.selectbox("Pilih game:", game_list)

# Jalankan rekomendasi jika ada game dipilih
if selected_game:
    # Cari index game
    example_idx = df_names[df_names == selected_game].index[0]

    # Cari neighbors
    distances, indices = nn_model.kneighbors([X_scaled[example_idx]])

    # Tampilkan info game asli
    st.subheader(f"Game Asli: {selected_game}")
    st.write(df.iloc[example_idx][['Platform', 'Genre', 'Global_Sales']])

    # Siapkan data rekomendasi
    recs = []
    for idx, dist in zip(indices[0][1:], distances[0][1:]):  # skip self
        recs.append({
            'Name': df.iloc[idx]['Name'],
            'Platform': df.iloc[idx]['Platform'],
            'Genre': df.iloc[idx]['Genre'],
            'Global_Sales': df.iloc[idx]['Global_Sales'],
            'Distance': dist
        })

    recs_df = pd.DataFrame(recs)

    # Tampilkan tabel rekomendasi
    st.subheader("Rekomendasi Game Serupa:")
    st.dataframe(recs_df)

    # Visualisasi plotly
    fig = px.bar(
        recs_df,
        x='Name',
        y='Global_Sales',
        color='Genre',
        hover_data=['Platform', 'Distance'],
        title=f"Rekomendasi Game Serupa dengan '{selected_game}'"
    )
    st.plotly_chart(fig)
