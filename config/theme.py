"""
Theme configuration — loads CSS from assets/ and manages dark/light toggle
"""

import streamlit as st
import os

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")


def get_current_theme():
    """Get the current theme from session state, default to dark"""
    if "app_theme" not in st.session_state:
        st.session_state.app_theme = "🌙 Dark"
    return st.session_state.app_theme


def apply_theme():
    """Load and apply the selected CSS theme"""
    theme = get_current_theme()

    if theme == "🌙 Dark":
        css_file = os.path.join(ASSETS_DIR, "styles.css")
    else:
        css_file = os.path.join(ASSETS_DIR, "styles_light.css")

    try:
        with open(css_file, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Theme file not found. Using default Streamlit theme.")


def is_dark_theme():
    """Check if the current theme is dark"""
    return get_current_theme() == "🌙 Dark"
