import streamlit as st
import base64
import os

def set_page_config():
    st.set_page_config(
        page_title="Activation Functions | Neural Studio",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def apply_custom_css():
    st.markdown("""
        <style>
        /* ── Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@600;700;800&display=swap');

        /* ── Design Tokens ── */
        :root {
            --stone-50:  #f9f8f6;
            --stone-100: #f0ede8;
            --stone-200: #e3ddd5;
            --stone-300: #cec7bc;
            --stone-400: #a8a094;
            --stone-500: #7a726a;
            --stone-900: #1a1714;
            --amber:     #c8902a;
            --amber-glow:rgba(200,144,42,0.25);
            --card-bg:   rgba(255,255,255,0.72);
            --border:    rgba(0,0,0,0.08);
            --shadow:    0 2px 24px rgba(0,0,0,0.06);
            --shadow-hover: 0 8px 40px rgba(0,0,0,0.12);
            --radius:    14px;
            --font-body: 'Inter', sans-serif;
            --font-head: 'Syne', sans-serif;
        }

        /* ── Reset & Base ── */
        html, body, [class*="css"] {
            font-family: var(--font-body) !important;
            color: chocolate !important;
        }
        
        /* ── Formulas ── */
        .katex, .katex-html {
            color: chocolate !important;
        }

        /* ── App Background ── */
        .stApp {
            background-color: var(--stone-100) !important;
            background-image:
                radial-gradient(ellipse 70% 50% at 20% 80%, rgba(200,144,42,0.06), transparent),
                radial-gradient(ellipse 60% 40% at 80% 20%, rgba(160,130,100,0.06), transparent);
        }

        /* ── Hide Streamlit Branding & Default Page Nav ── */
        #MainMenu, footer, header { visibility: hidden; }
        .stDeployButton { display: none; }
        [data-testid="stSidebarNav"] { display: none !important; }

        /* ── Main Content Padding ── */
        .block-container {
            padding: 2.5rem 3rem 4rem 3rem !important;
            max-width: 1200px !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: var(--stone-50) !important;
            border-right: 1px solid var(--border) !important;
            backdrop-filter: none !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--stone-900) !important;
        }

        /* ── Sidebar Brand ── */
        .sidebar-brand {
            font-family: var(--font-head);
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: var(--stone-400) !important;
            padding: 0.75rem 0 1.25rem 0;
        }

        /* ── Sidebar nav-link overrides ── */
        [data-testid="stSidebar"] .nav-link {
            border-radius: 8px !important;
            font-size: 0.85rem !important;
            color: var(--stone-500) !important;
            letter-spacing: 0.02em;
        }
        [data-testid="stSidebar"] .nav-link-selected {
            background-color: var(--stone-200) !important;
            color: var(--stone-900) !important;
            border-left: 3px solid var(--amber) !important;
            font-weight: 600 !important;
        }
        [data-testid="stSidebar"] .nav-link:hover {
            background-color: var(--stone-100) !important;
        }

        /* ── XP Badge in sidebar ── */
        .xp-badge {
            display: inline-block;
            background: var(--stone-900);
            color: chocolate !important;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.06em;
            padding: 4px 12px;
            border-radius: 99px;
            margin-top: 0.5rem;
        }

        /* ── Page Section Label ── */
        .section-label {
            font-family: var(--font-body);
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--stone-400);
            margin-bottom: 0.5rem;
        }

        /* ── Big Page Headings ── */
            h1, .markdown-text-container h1 {
            font-family: var(--font-head) !important;
            font-size: 3rem !important;
            font-weight: 800 !important;
            line-height: 1.08 !important;
            letter-spacing: -0.01em !important;
            text-transform: uppercase;
            color: chocolate !important;
            background: none !important;
            -webkit-text-fill-color: chocolate !important;
            margin-bottom: 0.75rem !important;
        }

        h2, .markdown-text-container h2 {
            font-family: var(--font-head) !important;
            font-size: 1.9rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            color: chocolate !important;
            background: none !important;
            -webkit-text-fill-color: chocolate !important;
        }

        h3, .markdown-text-container h3 {
            font-family: var(--font-head) !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: chocolate !important;
            background: none !important;
            -webkit-text-fill-color: chocolate !important;
        }

        p, li {
            color: chocolate !important;
            line-height: 1.7;
            font-size: 0.92rem;
        }

        /* ── Glass Cards ── */
        .glass-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.75rem;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            box-shadow: var(--shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .glass-card:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-3px);
            border-color: rgba(200,144,42,0.2);
        }

        /* ── Service Cards (3-col grid) ── */
        .service-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 2rem;
            box-shadow: var(--shadow);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .service-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, transparent, var(--amber), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .service-card:hover::before { opacity: 1; }

        .service-card:hover {
            box-shadow: var(--shadow-hover);
            transform: translateY(-4px);
        }

        .service-icon {
            width: 56px; height: 56px;
            border-radius: 12px;
            background: var(--stone-100);
            border: 1px solid var(--border);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }

        /* ── Divider ── */
        .page-divider {
            height: 1px;
            background: var(--border);
            margin: 2rem 0;
        }

        /* ── Pill Buttons ── */
        .stButton > button {
            background: var(--stone-900) !important;
            color: chocolate !important;
            border: none !important;
            border-radius: 99px !important;
            padding: 0.55rem 1.5rem !important;
            font-family: var(--font-body) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.06em !important;
            text-transform: uppercase !important;
            transition: all 0.25s ease !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.18) !important;
        }

        .stButton > button:hover {
            background: var(--amber) !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(200,144,42,0.35) !important;
        }


        /* ── Sliders ── */
        .stSlider > div > div > div > div {
            background-color: var(--stone-900) !important;
        }
        .stSlider [data-baseweb="slider"] [role="slider"] {
            background-color: var(--amber) !important;
            border-color: var(--amber) !important;
        }

        /* ── Select / Dropdowns ── */
        .stSelectbox [data-baseweb="select"] > div {
            background-color: var(--card-bg) !important;
            border-color: var(--border) !important;
            border-radius: 10px !important;
        }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: var(--stone-200);
            padding: 4px;
            border-radius: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border: none;
            border-radius: 8px;
            padding: 6px 18px;
            color: var(--stone-500);
            font-size: 0.82rem;
            font-weight: 500;
            letter-spacing: 0.04em;
        }

        .stTabs [aria-selected="true"] {
            background-color: chocolate;
            color: var(--stone-900) !important;
            font-weight: 700;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
        }

        /* ── Metrics ── */
        [data-testid="stMetricValue"] {
            font-family: var(--font-head) !important;
            font-size: 2rem !important;
            color: var(--stone-900) !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            color: var(--stone-400) !important;
        }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background-color: var(--card-bg);
            border-radius: 10px;
            font-family: var(--font-head);
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--stone-900);
            border: 1px solid var(--border);
        }

        /* ── Info / Warning boxes ── */
        .stAlert {
            background-color: var(--card-bg) !important;
            border-radius: 10px !important;
            border: 1px solid var(--border) !important;
            color: var(--stone-500) !important;
        }

        /* ── Dataframe ── */
        [data-testid="stDataFrame"] {
            border-radius: 10px !important;
            overflow: hidden;
            border: 1px solid var(--border) !important;
        }

        /* ── Horizontal Rule ── */
        hr {
            border-color: var(--border) !important;
            margin: 1.5rem 0 !important;
        }

        /* ── Animations ── */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes amber-pulse {
            0%   { box-shadow: 0 0 0   rgba(200,144,42,0); }
            50%  { box-shadow: 0 0 30px rgba(200,144,42,0.2); }
            100% { box-shadow: 0 0 0   rgba(200,144,42,0); }
        }

        .animate-up    { animation: fadeUp 0.6s ease both; }
        .animate-glow  { animation: amber-pulse 3s ease-in-out infinite; }
        .delay-1       { animation-delay: 0.15s; }
        .delay-2       { animation-delay: 0.30s; }
        .delay-3       { animation-delay: 0.45s; }

        /* ── Hero counter label ── */
        .counter-label {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: var(--stone-400);
        }

        </style>
    """, unsafe_allow_html=True)


def card(title, content, icon=""):
    st.markdown(f"""
        <div class="glass-card animate-up">
            <div class="service-icon">{icon}</div>
            <h3 style="margin-top:0; margin-bottom: 0.4rem;">{title}</h3>
            <p style="margin-bottom:0;">{content}</p>
        </div>
    """, unsafe_allow_html=True)
