  # frontend/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import os

CSV_PATH = "../backend/data/tdnet_score_all.csv"

st.set_page_config(page_title="TDnet Analyzer", layout="wide")

st.title("📊 TDnet 開示情報 スコアランキング")

# ファイル読み込み
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    st.success(f"開示件数: {len(df)} 件を読み込みました。")

    # フィルター
    col1, col2, col3 = st.columns(3)
    with col1:
        genre_filter = st.selectbox("ジャンルで絞り込み", ["全て"] + sorted(df["genre"].dropna().unique().tolist()))
    with col2:
        sentiment_filter = st.selectbox("ポジネガで絞り込み", ["全て", "ポジティブ", "ネガティブ", "中立"])
    with col3:
        score_sort = st.selectbox("並び順", ["スコア順", "スコア逆順"])

    filtered = df.copy()
    if genre_filter != "全て":
        filtered = filtered[filtered["genre"] == genre_filter]
    if sentiment_filter != "全て":
        filtered = filtered[filtered["sentiment"] == sentiment_filter]
    if score_sort == "スコア逆順":
        filtered = filtered.sort_values("スコア", ascending=True)
    else:
        filtered = filtered.sort_values("スコア", ascending=False)

    st.dataframe(filtered, use_container_width=True)

    # スコアヒートマップ
    fig = px.bar(filtered.head(20), x="title", y="スコア", color="genre", title="スコア上位 20 件")
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("スコアCSVが見つかりません。先に backend/batch.py を実行してください。")
