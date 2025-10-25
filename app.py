import streamlit as st
import tempfile
import base64
from pathlib import Path
from analyzer import process_video

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
        # 画像がない場合は何もしない（またはデフォルト色を設定）
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

st.title("Motion Analyzer")

with st.sidebar:
    st.title("FOR ALL SKIERS")
    st.caption("Visualization of Skeletal Structure and Joint Angle Variations")

    # 背景選択（ラジオボタン）
    background_option = st.radio("Background", ["Show original video", "Hide background"])

    # 関節角度選択肢
    all_angles = [
        "Knee Ext/Flex",
        "Knee Abd/Add",
        "Hip Ext/Flex",
        "Hip Abd/Add",
        "Torso Tilt",
        "Inclination Angle"
    ]

    select_all = st.checkbox("Select all joint angles", value=True)
    if select_all:
        angle_options = all_angles
    else:
        angle_options = st.multiselect(
            "Joint angles to display",
            all_angles,
            default=["Knee Ext/Flex", "Hip Ext/Flex", "Torso Tilt"]
        )

uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov"])
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
            f"<p style='text-align:center; color:#007BFF; font-size:18px;'>ANALYZING... {percent}/100% </p>",
            unsafe_allow_html=True
        )

    output_path = process_video(
        input_path,
        progress_callback=update_progress,
        show_background=(background_option == "Show original video"),
        selected_angles=angle_options
    )

    progress_container.markdown(
        """
        <div style="width: 100%; background-color: white; height: 20px; border-radius: 5px;"></div>
        """,
        unsafe_allow_html=True
    )
    progress_text.markdown(
        "<p style='text-align:center; color:white; font-size:20px;'>Successfully done</p>",
        unsafe_allow_html=True
    )

    with open(output_path, "rb") as f:
        video_bytes = f.read()
        st.download_button(
            label="Download here",
            data=video_bytes,
            file_name="analyzed_ski_video.mp4",
            mime="video/mp4"

        )






