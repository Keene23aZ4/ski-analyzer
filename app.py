import streamlit as st
import tempfile
import base64
from pathlib import Path
from analyzer import process_video

# 言語選択
language = st.sidebar.selectbox("Language / 言語", ["English", "日本語"])

# 翻訳辞書
translations = {
    "English": {
        "title": "Motion Analyzer",
        "sidebar_title": "FOR ALL SKIERS",
        "caption": "Visualization of Skeletal Structure and Joint Angle Variations",
        "background": "Background",
        "background_options": ["Show original video", "Hide background"],
        "select_all": "Select all joint angles",
        "angle_prompt": "Joint angles to display",
        "upload": "Upload your video",
        "analyzing": "ANALYZING... {percent}/100%",
        "done": "Successfully done",
        "download": "Download here"
    },
    "日本語": {
        "title": "滑走動作分析システム",
        "sidebar_title": "",
        "caption": "できること　骨格構造、関節角度、前傾角度、内傾角度の変化を可視化",
        "background": "動画の背景の有無を選択",
        "background_options": ["表示", "非表示"],
        "select_all": "すべての関節角度を選択する",
        "angle_prompt": "表示する関節角度",
        "upload": "動画をアップロード",
        "analyzing": "解析中... {percent}/100%",
        "done": "解析完了",
        "download": "ダウンロード"
    }
}
t = translations[language]

# 背景画像設定
def set_background():
    img_path = Path(__file__).parent / "static" / "1704273575813.jpg"
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

# タイトル
st.title(t["title"])

# サイドバー
with st.sidebar:
    st.title(t["sidebar_title"])
    st.caption(t["caption"])

    background_option = st.radio(t["background"], t["background_options"])

    all_angles = [
        "Knee Ext/Flex",
        "Knee Abd/Add",
        "Hip Ext/Flex",
        "Hip Abd/Add",
        "Torso Tilt",
        "Inclination Angle"
    ]

    select_all = st.checkbox(t["select_all"], value=True)
    if select_all:
        angle_options = all_angles
    else:
        angle_options = st.multiselect(
            t["angle_prompt"],
            all_angles,
            default=["Knee Ext/Flex", "Hip Ext/Flex", "Torso Tilt"]
        )

# ファイルアップロード
uploaded_file = st.file_uploader(t["upload"], type=["mp4", "mov"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    progress_container = st.empty()
    progress_text = st.empty()

    def update_progress(progress):
        percent = int(progress * 100)
        progress_container.markdown(
            f"""
            <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px;">
                <div style="width: {percent}%; background-color: #007BFF; height: 20px; border-radius: 5px;"></div>
            </div>
            """,
            unsafe_allow_html=True
        )
        progress_text.markdown(
            f"<p style='text-align:center; color:#007BFF; font-size:18px;'>{t['analyzing'].format(percent=percent)}</p>",
            unsafe_allow_html=True
        )

    output_path = process_video(
        input_path,
        progress_callback=update_progress,
        show_background=(background_option == t["background_options"][0]),
        selected_angles=angle_options
    )

    progress_container.markdown(
        """
        <div style="width: 100%; background-color: white; height: 20px; border-radius: 5px;"></div>
        """,
        unsafe_allow_html=True
    )
    progress_text.markdown(
        f"<p style='text-align:center; color:white; font-size:20px;'>{t['done']}</p>",
        unsafe_allow_html=True
    )

    with open(output_path, "rb") as f:
        video_bytes = f.read()
        st.download_button(
            label=t["download"],
            data=video_bytes,
            file_name="analyzed_ski_video.mp4",
            mime="video/mp4"
        )



