import streamlit as st
import plotly.graph_objects as go
import numpy as np
import base64, os

# ── helper: encode local image ────────────────────────────────────
def _img_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def create_sparkline(func, title, color="#1a1714"):
    x = np.linspace(-5, 5, 120)
    y = func(x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines',
        line=dict(color=color, width=2.5),
        fill='tozeroy',
        fillcolor='rgba(200,144,42,0.08)'
    ))
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(color='#1a1714', size=13, family='Syne'),
            x=0.04
        ),
        height=200,
        margin=dict(l=8, r=8, t=36, b=8),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, zeroline=True,
                   zerolinecolor='rgba(0,0,0,0.1)', showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=True,
                   zerolinecolor='rgba(0,0,0,0.1)', showticklabels=False),
    )
    return fig

def render():
    # ── Hero ─────────────────────────────────────────────────────
    orb_path = os.path.join("assets", "hero_orb.png")
    orb_b64  = _img_b64(orb_path)
    orb_tag  = f'<img src="data:image/png;base64,{orb_b64}" style="width:100%;max-width:480px;opacity:0.92;" />' if orb_b64 else ""

    col_text, col_orb = st.columns([1, 1])

    with col_text:
        st.markdown("""
            <div style="padding: 3.5rem 0 2rem 0;" class="animate-up">
                <div class="section-label">01 / 10 &nbsp;·&nbsp; Welcome</div>
                <h1>Activation<br>Functions,<br>Explained</h1>
                <p style="max-width:420px; font-size:0.97rem; margin-bottom:2rem;">
                    Understand how AI neurons think, activate, and learn —
                    through interactive visuals and hands-on experimentation.
                </p>
            </div>
        """, unsafe_allow_html=True)

        btn_col1, btn_col2, _ = st.columns([1, 1, 2])
        with btn_col1:
            if st.button("Get Started", key="hero_start"):
                pass
        with btn_col2:
            if st.button("Our Modules", key="hero_modules"):
                pass

    with col_orb:
        if orb_tag:
            st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:center;
                            padding:2rem 0; min-height:320px;" class="animate-glow">
                    {orb_tag}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="display:flex; align-items:center; justify-content:center;
                            min-height:320px; font-size:8rem;">
                    🧠
                </div>
            """, unsafe_allow_html=True)

    # ── Divider ───────────────────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    # ── Section: Discover ─────────────────────────────────────────
    st.markdown("""
        <div class="animate-up delay-1">
            <div class="section-label">02 / 10 &nbsp;·&nbsp; Overview</div>
            <h2 style="margin-bottom:0.4rem;">Discover<br>Our Modules</h2>
            <p style="max-width:380px; margin-bottom:2rem;">
                We deliver complete activation-function education under one roof.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── 3-column service cards ────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
            <div class="service-card animate-up delay-1">
                <div class="service-icon">⚡</div>
                <h3>Playground</h3>
                <p>Interact with activation functions in real-time. Tweak inputs, view outputs, explore edge cases.</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
            <div class="service-card animate-up delay-2">
                <div class="service-icon">⚔️</div>
                <h3>Battle Arena</h3>
                <p>Compare two activation functions head-to-head on custom datasets and see which wins.</p>
            </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
            <div class="service-card animate-up delay-3">
                <div class="service-icon">🏗️</div>
                <h3>Build Your Net</h3>
                <p>Configure your own neural network layer-by-layer, then train it on sample datasets.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Sparkline Preview Row ─────────────────────────────────────
    st.markdown("""
        <div class="animate-up delay-2">
            <div class="section-label">Key Functions</div>
            <h2 style="margin-bottom:1.5rem;">Quick Preview</h2>
        </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3, gap="medium")

    funcs = [
        (lambda x: np.maximum(0, x),      "ReLU"),
        (lambda x: 1/(1+np.exp(-x)),       "Sigmoid"),
        (lambda x: np.tanh(x),             "Tanh"),
    ]
    for col, (fn, name) in zip([sc1, sc2, sc3], funcs):
        with col:
            st.markdown('<div class="glass-card" style="padding:1rem;">', unsafe_allow_html=True)
            st.plotly_chart(
                create_sparkline(fn, name),
                use_container_width=True,
                config={'displayModeBar': False}
            )
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Info strip ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card animate-up" style="display:flex; align-items:center; gap:1.5rem; padding:1.25rem 2rem;">
            <div style="font-size:1.6rem;">👈</div>
            <div>
                <div class="section-label" style="margin-bottom:0.1rem;">Navigation</div>
                <p style="margin:0; font-size:0.88rem;">Use the sidebar to explore all 10 sections — earn XP and badges as you go!</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
