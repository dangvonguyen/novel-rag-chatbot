import streamlit as st


def render_header(header: str = "MindEase"):
    body = """<div class="header"><h3>{header}</h3></div>""".format(header=header)
    st.markdown(body, unsafe_allow_html=True)
