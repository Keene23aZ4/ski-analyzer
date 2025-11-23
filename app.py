import streamlit as st
import tempfile
import base64
from pathlib import Path
from analyzer import process_video

# 言語選択
language = st.sidebar.selectbox("Language / 言語 /语言 /語言/ 언어", ["English", "日本語", "简体中文", "繁体中文", "한국어"])

# 翻訳辞書
translations = {
    "English": {
        "title": "Motion Analyzer",
        "sidebar_title": "FOR ALL SKIERs",
        "caption": "Visualization of Skeletal Structure, Joint Angle, Inclination Angle and Torso Tilt Angle",
        "background": "Background",
        "background_options": ["Show original video", "Hide background"],
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
        "title": "滑走動作分析システム",
        "sidebar_title": "",
        "caption": "骨格構造、関節角度、前傾角度、内傾角度の変化を可視化",
        "background": "動画の背景の有無を選択",
        "background_options": ["表示", "非表示"],
        "select_all": "すべての関節角度を選択する",
        "angle_prompt": "表示する関節角度",
        "upload": "動画をアップロード",
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
        "caption": "可视化骨骼结构、关节角度、前倾角度和内倾角度的变化",
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
    },
    "繁体中文": {
        "title": "滑雪動作分析應用程式",
        "sidebar_title": "致所有滑雪者",
        "caption": "可視化骨骼結構、關節角度、前傾角度與內傾角度的變化",
        "background": "選擇是否顯示影片背景",
        "background_options": ["顯示", "隱藏"],
        "select_all": "選擇所有關節角度",
        "angle_prompt": "選擇要顯示的關節角度",
        "upload": "上傳影片",
        "analyzing": "分析中... {percent}/100%",
        "done": "分析完成",
        "download": "下載",
        "Knee Ext/Flex": "膝關節 伸展／屈曲",
        "Knee Abd/Add": "膝關節 外展／內收",
        "Hip Ext/Flex": "髖關節 伸展／屈曲",
        "Hip Abd/Add": "髖關節 外展／內收",
        "Torso Tilt": "前傾角度",
        "Inclination Angle": "內傾角度"
    },
    "한국어": {
        "title": "스키 동작 분석 애플리케이션",
        "sidebar_title": "모든 스키어를 위해",
        "caption": "골격 구조, 관절 각도, 전경 각도, 내경 각도의 변화를 시각화",
        "background": "영상 배경 표시 여부 선택",
        "background_options": ["표시", "숨김"],
        "select_all": "모든 관절 각도 선택",
        "angle_prompt": "표시할 관절 각도 선택",
        "upload": "영상 업로드",
        "analyzing": "분석 중... {percent}/100%",
        "done": "분석 완료",
        "download": "다운로드",
        "Knee Ext/Flex": "무릎 관절 신전/굴곡",
        "Knee Abd/Add": "무릎 관절 외전/내전",
        "Hip Ext/Flex": "엉덩이 관절 신전/굴곡",
        "Hip Abd/Add": "엉덩이 관절 외전/내전",
        "Torso Tilt": "전경 각도",
        "Inclination Angle": "내경 각도"
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


















