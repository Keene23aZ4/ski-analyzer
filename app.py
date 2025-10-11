import streamlit as st
import tempfile
from analyzer import process_video  # analyzer.py に定義された関数を使う

st.title("スキー動作解析アプリ")

uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    output_path = process_video(input_path)
    st.video(output_path)