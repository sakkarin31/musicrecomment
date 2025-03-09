import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from annoy import AnnoyIndex  # ใช้ Annoy
import joblib
import ast
import os

# ฟังก์ชันเพิ่มพื้นหลัง
def add_bg_from_url():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1511671782779-c97d3d27a1d4");
            background-size: cover;
            color: #ffffff;
        }
        .title-text {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #ff4b4b;
        }
        .subtitle-text {
            text-align: center;
            font-size: 20px;
            color: #ffffff;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: #ffffff;
            border-radius: 12px;
            padding: 8px 24px;
            font-size: 18px;
        }
        .song-card {
            background-color: rgba(0, 0, 0, 0.7);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()


@st.cache_resource
def load_models():
    autoencoder = load_model("autoencoder_model.h5", custom_objects={"mse": "mean_squared_error"})
    encoder = load_model("encoder_model.h5")
    scaler = joblib.load("scaler.pkl")
    return autoencoder, encoder, scaler


@st.cache_data
def load_data():
    df = pd.read_csv("tracks_features.csv")
    features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'valence', 'tempo']
    df_clean = df.dropna(subset=features)
    X = df_clean[features].values
    return df_clean, X


# โหลดโมเดลและข้อมูล
autoencoder, encoder, scaler = load_models()
df_clean, X = load_data()

X_scaled = scaler.transform(X)

# เช็คว่าไฟล์ X_encoded.npy มีอยู่หรือไม่
if os.path.exists("X_encoded.npy"):
    X_encoded = np.load("X_encoded.npy")
else:
    X_encoded = encoder.predict(X_scaled)
    np.save("X_encoded.npy", X_encoded)  # เก็บค่า X_encoded ลงไฟล์ .npy

df_clean["vector"] = list(X_encoded)
X_encoded_array = np.array(X_encoded)

# สร้าง Annoy Index
f = X_encoded_array.shape[1]  # จำนวนมิติของเวกเตอร์
annoy_index = AnnoyIndex(f, 'angular')  # ใช้ 'angular' สำหรับ cosine distance
for i, vector in enumerate(X_encoded_array):
    annoy_index.add_item(i, vector)
annoy_index.build(10)  # สร้างต้นไม้ 10 ต้น


def recommend_similar(song_name, artist_name, df, top_n=5):
    song = df[(df["name"].str.lower() == song_name.lower()) & 
              (df["artists"].apply(lambda x: artist_name.lower() in [a.lower() for a in eval(x)]))]

    if song.empty:
        return "ไม่พบเพลงนี้ใน Dataset"

    song_vector = np.array(song["vector"].iloc[0]).reshape(1, -1)

    # ค้นหาความคล้ายคลึงโดยใช้ Annoy
    indices = annoy_index.get_nns_by_vector(song_vector.flatten(), top_n * 2, include_distances=False)
    recommended_songs = df.iloc[indices]
    recommended_songs = recommended_songs[~((recommended_songs["name"].str.lower() == song_name.lower()) &
                                            (recommended_songs["artists"].apply(lambda x: artist_name.lower() in [a.lower() for a in eval(x)])))] 

    # กรองเพลงที่แนะนำ
    seen_songs = set()
    unique_recommendations = []
    for _, song in recommended_songs.iterrows():
        song_id = (song["name"].lower(), tuple(sorted(eval(song["artists"]))))  # ทำให้ศิลปินเรียงลำดับเพื่อป้องกันความซ้ำซ้อน
        if song_id not in seen_songs:
            seen_songs.add(song_id)
            unique_recommendations.append(song)
        if len(unique_recommendations) == top_n:
            break

    if not unique_recommendations:
        return "ไม่พบเพลงแนะนำที่ตรงตามเงื่อนไข"

    return pd.DataFrame(unique_recommendations)[["name", "artists", "album"]]


# Streamlit UI
st.markdown(
    """
    <div style="
        background-color: rgba(0, 0, 0, 0.8); 
        padding: 20px; 
        border-radius: 15px; 
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5); 
        margin-bottom: 20px;
    ">
        <h1 style="color: #ff4b4b; font-size: 50px;">🎵 Match this song 🎵</h1>
        <p style="color: #ffffff; font-size: 20px;">ค้นหาเพลงใหม่ๆ ที่ตรงใจคุณได้ง่ายๆ!</p>
    </div>
    """,
    unsafe_allow_html=True
)

song_name = st.text_input("🔍 พิมพ์ชื่อเพลงที่คุณต้องการแนะนำ:")
artist_name = st.text_input("🎤 พิมพ์ชื่อศิลปิน (ถ้ามี):")

if st.button("ค้นหาเพลงแนะนำ"):
    if song_name.strip() == "" or artist_name.strip() == "":
        st.warning("⚠️ กรุณากรอกชื่อเพลงและศิลปินให้ครบถ้วน")
    else:
        result = recommend_similar(song_name, artist_name, df_clean)
        if isinstance(result, str):
            st.error(result)
        else:
            st.success("🎧 แนะนำเพลงที่คล้ายกัน:")
            
            # แสดงผลในรูปแบบการ์ด
            for index, row in result.iterrows():
                st.markdown(
                    f"""
                    <div class="song-card">
                        <h4>🎵 {row['name']}</h4>
                        <p>👨‍🎤 ศิลปิน: {", ".join(ast.literal_eval(row['artists']))}</p>
                        <p>💿 อัลบั้ม: {row['album']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
