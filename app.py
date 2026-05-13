import streamlit as st
from streamlit_option_menu import option_menu
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.ui import set_page_config, apply_custom_css

set_page_config()
apply_custom_css()

from pages import (
    p01_Home, p02_Neural_Basics, p03_Playground, p04_Why_It_Matters,
    p05_Battle_Arena, p06_Gradient_Flow, p07_Real_Applications,
    p08_Build_Your_Net, p09_Challenge_Zone, p10_Cheatsheet
)

# ── Session State ────────────────────────────────────────────────
if 'xp'     not in st.session_state: st.session_state.xp     = 0
if 'badges' not in st.session_state: st.session_state.badges = []
if 'score'  not in st.session_state: st.session_state.score  = 0

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">NEURAL STUDIO</div>
        <div style="height:1px; background:var(--border); margin-bottom:1.25rem;"></div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Neural Basics",
            "Playground",
            "Why It Matters",
            "Battle Arena",
            "Gradient Flow",
            "Real Applications",
            "Build Your Net",
            "Challenge Zone",
            "Cheatsheet",
        ],
        icons=[
            "house-fill",
            "diagram-2-fill",
            "lightning-fill",
            "question-circle-fill",
            "shield-fill",
            "water",
            "globe2",
            "bricks",
            "controller",
            "file-earmark-text-fill",
        ],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "0 !important",
                "background-color": "transparent",
            },
            "icon": {
                "color": "var(--stone-400)",
                "font-size": "0.95rem",
            },
            "nav-link": {
                "font-size": "0.82rem",
                "font-weight": "500",
                "text-align": "left",
                "margin": "1px 0",
                "padding": "8px 12px",
                "border-radius": "8px",
                "font-family": "Inter, sans-serif",
                "color": "var(--stone-500)",
                "letter-spacing": "0.01em",
                "--hover-color": "var(--stone-100)",
            },
            "nav-link-selected": {
                "background-color": "var(--stone-200)",
                "color": "var(--stone-900)",
                "border-left": "3px solid var(--amber)",
                "font-weight": "700",
            },
        },
    )

    # XP & Badges
    st.markdown("<div style='height:1px; background:var(--border); margin: 1.25rem 0;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="counter-label">Progress</div>
        <span class="xp-badge">⭐ {st.session_state.xp} XP</span>
    """, unsafe_allow_html=True)

    if st.session_state.badges:
        st.markdown(
            "<div style='margin-top:0.5rem; font-size:1.1rem;'>" +
            " ".join(st.session_state.badges) +
            "</div>",
            unsafe_allow_html=True
        )

# ── Routing ──────────────────────────────────────────────────────
routes = {
    "Home":              p01_Home,
    "Neural Basics":     p02_Neural_Basics,
    "Playground":        p03_Playground,
    "Why It Matters":    p04_Why_It_Matters,
    "Battle Arena":      p05_Battle_Arena,
    "Gradient Flow":     p06_Gradient_Flow,
    "Real Applications": p07_Real_Applications,
    "Build Your Net":    p08_Build_Your_Net,
    "Challenge Zone":    p09_Challenge_Zone,
    "Cheatsheet":        p10_Cheatsheet,
}

routes[selected].render()
