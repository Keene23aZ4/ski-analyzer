import streamlit as st
import tempfile
import base64
from pathlib import Path
from analyzer import process_video


st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">',
    unsafe_allow_html=True
)

# CSSを文字列として埋め込む
st.markdown(
    """
    <style>
    h1, p, div, section[data-testid="stSidebar"] {
        font-family: 'Press Start 2P', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 翻訳辞書
translations = {
    "English": {
        "title": "Motion Analyzer",
        "sidebar_title": "FOR ALL SKIERs",
        "background": "Background",
        "background_options": ["Show", "Hide"],
        "select_all": "Select all joint angles",
        "angle_prompt": "Joint angles to display",
        "upload": "Upload your video",
        "analyzing": "ANALYZING... {percent}/100%",
        "done": "Successfully done",
        "download": "Download here",
        "Knee Ext/Flex": "Knee Ext/Flex",
        "Knee Abd/Add": "Knee Abd/Add",
        "Hip Ext/Flex": "Hip Ext/Flex",
        "Hip Abd/Add": "Hip Abd/Add",
        "Torso Tilt": "Torso Tilt",
        "Inclination Angle": "Inclination Angle"
    },
    "日本語": {
        "title": "滑走分析アプリ",
        "sidebar_title": "",
        "background": "動画の背景",
        "background_options": ["表示", "非表示 (骨格のみ抽出)"],
        "select_all": "すべての関節角度を選択する",
        "angle_prompt": "表示する関節角度",
        "upload": "動画をアップロード (5.00MB以下推奨)",
        "analyzing": "解析中... {percent}/100%",
        "done": "解析完了",
        "download": "ダウンロード",
        "Knee Ext/Flex": "膝関節 伸展・屈曲",
        "Knee Abd/Add": "膝関節 内外転",
        "Hip Ext/Flex": "股関節 伸展・屈曲",
        "Hip Abd/Add": "股関節 内外転",
        "Torso Tilt": "前傾角度",
        "Inclination Angle": "内傾角度"
    },
    "简体中文": {
        "title": "滑雪动作分析应用",
        "sidebar_title": "致所有滑雪者",
        "background": "选择是否显示视频背景",
        "background_options": ["显示", "隐藏"],
        "select_all": "选择所有关节角度",
        "angle_prompt": "选择要显示的关节角度",
        "upload": "上传视频",
        "analyzing": "分析中... {percent}/100%",
        "done": "分析完成",
        "download": "下载",
        "Knee Ext/Flex": "膝关节 伸展/屈曲",
        "Knee Abd/Add": "膝关节 外展/内收",
        "Hip Ext/Flex": "髋关节 伸展/屈曲",
        "Hip Abd/Add": "髋关节 外展/内收",
        "Torso Tilt": "前倾角度",
        "Inclination Angle": "内倾角度"
    }
}
lang = st.session_state.get("language", "English")
t = translations[lang]


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
# 背景画像設定
st.set_page_config(page_title="top page", page_icon="")

# タイトル
st.title(t["title"])

# サイドバー
with st.sidebar:
    st.title(t["sidebar_title"])


    background_option = st.radio(t["background"], t["background_options"])

    

import streamlit as st

# CSSで input[type=file] を隠す
st.markdown(
    """
    <style>
    input[type="file"] {
        display: none;
    }
    .custom-upload {
        display: inline-block;
        padding: 10px 20px;
        background-color: #007BFF;
        color: white;
        font-family: 'Press Start 2P', monospace;
        font-size: 14px;
        border-radius: 5px;
        cursor: pointer;
        text-align: center;
    }
    .custom-upload:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# file_uploader を通常通り配置
uploaded_file = st.file_uploader(t["upload"], type=["mp4", "mov"])
# カスタムボタンをラベルとして表示
st.markdown('<label class="custom-upload">🎮 Browse File<input type="file"></label>', unsafe_allow_html=True)


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




















































































