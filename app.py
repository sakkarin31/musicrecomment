import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.spatial import KDTree
import joblib
import ast
import os

# โหลดโมเดลและข้อมูล
autoencoder = load_model("autoencoder_model.h5", custom_objects={"mse": "mean_squared_error"})
encoder = load_model("encoder_model.h5")
df = pd.read_csv("tracks_features.csv")

# เตรียมข้อมูล
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'valence', 'tempo']
df_clean = df.dropna(subset=features)
X = df_clean[features].values
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X)

# ตรวจสอบว่าไฟล์ X_encoded.npy มีอยู่หรือไม่
if os.path.exists("X_encoded.npy"):
    # หากมีไฟล์ ให้โหลด X_encoded จากไฟล์
    X_encoded = np.load("X_encoded.npy")
else:
    # หากไม่มีไฟล์ ให้คำนวณ X_encoded และบันทึกลงไฟล์
    X_encoded = encoder.predict(X_scaled)
    np.save("X_encoded.npy", X_encoded)

# เพิ่มคอลัมน์เวกเตอร์ไปยัง DataFrame
df_clean["vector"] = list(X_encoded)

# สร้าง KD-Tree
X_encoded_array = np.array(X_encoded)
kdtree = KDTree(X_encoded_array)

# เพิ่มคอลัมน์ index
df_clean = df_clean.reset_index()

# ฟังก์ชันแนะนำเพลงที่คล้ายกัน
def recommend_similar(song_name, artist_name, df, top_n=5):
    # ค้นหาเพลงที่ตรงกับชื่อเพลงและศิลปิน
    song = df[(df["name"].str.lower() == song_name.lower()) &
              (df["artists"].apply(lambda x: artist_name.lower() in [a.lower() for a in eval(x)]))]

    if song.empty:
        return "ไม่พบเพลงนี้ใน Dataset"

    # ดึงเวกเตอร์ของเพลงที่ค้นหา
    song_vector = np.array(song["vector"].iloc[0]).reshape(1, -1)

    # ค้นหาเพลงที่คล้ายกันด้วย KD-Tree
    distances, indices = kdtree.query(song_vector, k=top_n * 2)
    recommended_songs = df.iloc[indices.flatten()]

    # กรองเพลงที่เป็นเพลงต้นฉบับออก
    recommended_songs = recommended_songs[~((recommended_songs["name"].str.lower() == song_name.lower()) & 
                                            (recommended_songs["artists"].apply(lambda x: artist_name.lower() in [a.lower() for a in eval(x)])))]

    # เพิ่มการเก็บเพลงที่แนะนำไปแล้วเพื่อป้องกันการแนะนำซ้ำ
    seen_songs = set()
    unique_recommendations = []

    for _, song in recommended_songs.iterrows():
        song_id = (song["name"].lower(), tuple(sorted(eval(song["artists"])))) 
        if song_id not in seen_songs:
            seen_songs.add(song_id)
            unique_recommendations.append(song)
        if len(unique_recommendations) == top_n:
            break

    if not unique_recommendations:
        return "ไม่พบเพลงแนะนำที่ตรงตามเงื่อนไข"

    return pd.DataFrame(unique_recommendations)[["name", "artists", "album"]]

# ส่วนติดต่อผู้ใช้ด้วย Streamlit
st.title("🎶 ระบบแนะนำเพลงที่คล้ายกัน")

song_name = st.text_input("🔍 พิมพ์ชื่อเพลงที่คุณต้องการแนะนำ:")
artist_name = st.text_input("🎤 พิมพ์ชื่อศิลปิน (ถ้ามี):")

if st.button("ค้นหาเพลงแนะนำ"):
    if song_name.strip() == "" or artist_name.strip() == "":
        st.warning("กรุณากรอกชื่อเพลงและศิลปินให้ครบถ้วน")
    else:
        result = recommend_similar(song_name, artist_name, df_clean)
        if isinstance(result, str):
            st.error(result)
        else:
            st.success("🎧 แนะนำเพลงที่คล้ายกัน:")
            st.dataframe(result)
