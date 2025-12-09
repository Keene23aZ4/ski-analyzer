import streamlit as st
import tempfile
import base64
from pathlib import Path

font_path = Path(__file__).parent.parent / "static" / "BestTen-CRT.otf"
if font_path.exists():
    encoded = base64.b64encode(font_path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'BestTen';
            src: url(data:font/opentype;base64,{encoded}) format('opentype');
            font-display: swap;
        }}
        h1, p, div {{
            font-family: 'BestTen', monospace !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 背景画像設定
def set_background():
    img_path = Path("static/1704273575813.jpg")
    if img_path.exists():
        encoded = base64.b64encode(img_path.read_bytes()).decode()
        mime = "image/jpeg"
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:{mime};base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
set_background()
st.title("About")
translations = {
    "English": {
        "write": "coming soon......"
    },
    "日本語":{
        "write": "準備中"
    }
}
lang = st.session_state.get("language", "English")
t = translations[lang]
st.write(t["write"])

