import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# --- Load dataset ---
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('vgsales.csv')
    
    # Drop kolom tidak dipakai
    df_clean = df.drop(['Rank', 'Publisher'], axis=1)
    df_clean['Year'] = df_clean['Year'].fillna(df_clean['Year'].median())
    
    # Simpan nama game
    df_names = df_clean['Name']
    
    # Drop Name untuk fitur
    df_clean = df_clean.drop(['Name'], axis=1)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'])
    
    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    
    return df, df_names, X_scaled

# Load data sekali saat awal
df, df_names, X_scaled = load_and_process_data()

# --- Fit Nearest Neighbors model ---
@st.cache_resource
def train_model(X_scaled):
    model = NearestNeighbors(n_neighbors=6, metric='euclidean')
    model.fit(X_scaled)
    return model

nn_model = train_model(X_scaled)

# --- Streamlit UI ---
st.title("🎮 Sistem Rekomendasi Video Game Berdasarkan Data Penjualan")
st.write("Pilih game di bawah ini untuk mendapatkan rekomendasi game serupa berdasarkan penjualan dan fitur.")

# Dropdown pilih game
game_list = df_names.tolist()
selected_game = st.selectbox("Pilih game:", game_list)

# Saat game dipilih
if selected_game:
    # Cari index game
    example_idx = df_names[df_names == selected_game].index[0]
    
    # Cari neighbors
    distances, indices = nn_model.kneighbors([X_scaled[example_idx]])
    
    # Info game asli
    st.subheader(f"Game Asli: {selected_game}")
    st.write(df.iloc[example_idx][['Platform', 'Genre', 'Global_Sales']])
    
    # Kumpulkan rekomendasi
    recs = []
    for idx, dist in zip(indices[0][1:], distances[0][1:]):
        recs.append({
            'Name': df.iloc[idx]['Name'],
            'Platform': df.iloc[idx]['Platform'],
            'Genre': df.iloc[idx]['Genre'],
            'Global_Sales': df.iloc[idx]['Global_Sales'],
            'Distance': dist
        })
    
    recs_df = pd.DataFrame(recs)
    
    # Tabel rekomendasi
    st.subheader("Rekomendasi Game Serupa:")
    st.dataframe(recs_df)
    
    # Visualisasi
    fig = px.bar(
        recs_df,
        x='Name',
        y='Global_Sales',
        color='Genre',
        hover_data=['Platform', 'Distance'],
        title=f"Rekomendasi Game Serupa dengan '{selected_game}'"
    )
    st.plotly_chart(fig)
