import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import plotly.express as px

# --- Load dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv('content/vgsales.csv')
    df_clean = df.drop(['Rank', 'Publisher'], axis=1)
    df_clean['Year'] = df_clean['Year'].fillna(df_clean['Year'].median())
    df_names = df_clean['Name']
    df_clean = df_clean.drop(['Name'], axis=1)
    df_encoded = pd.get_dummies(df_clean, columns=['Platform', 'Genre'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    return df, df_names, X_scaled, df_encoded

df, df_names, X_scaled, df_encoded = load_data()

# --- Fit model ---
knn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn_model.fit(X_scaled)

# --- Streamlit UI ---
st.title("🎮 Sistem Rekomendasi Video Game Berdasarkan Data Penjualan")
st.write("Gunakan aplikasi ini untuk menemukan video game serupa berdasarkan penjualan dan fitur lainnya.")

# Pilih game
game_list = df_names.tolist()
selected_game = st.selectbox("Pilih game untuk melihat rekomendasi:", game_list)

# Cari index game terpilih
if selected_game:
    example_idx = df_names[df_names == selected_game].index[0]

    # Cari rekomendasi
    distances, indices = knn_model.kneighbors([X_scaled[example_idx]])

    # Tampilkan game asli
    st.subheader(f"Game Asli: {selected_game}")
    st.write(df.iloc[example_idx][['Platform', 'Genre', 'Global_Sales']])

    # Siapkan data rekomendasi
    recs = []
    for idx, dist in zip(indices[0][1:], distances[0][1:]):
        recs.append({
            'Name': df_names.iloc[idx],
            'Platform': df.iloc[idx]['Platform'],
            'Genre': df.iloc[idx]['Genre'],
            'Global_Sales': df.iloc[idx]['Global_Sales'],
            'Distance': dist
        })

    recs_df = pd.DataFrame(recs)
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
