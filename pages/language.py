import streamlit as st

languages = ["English", "日本語", "简体中文"]

st.title("Language Settings")
st.session_state["language"] = st.selectbox("Choose your language", languages)
