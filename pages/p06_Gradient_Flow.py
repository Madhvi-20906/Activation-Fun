import streamlit as st
import plotly.graph_objects as go
import numpy as np

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">06 / 10 &nbsp;·&nbsp; Deep Learning</div>
            <h1>Gradient<br>Flow Visualizer</h1>
            <p style="max-width:500px;">Watch gradients flow backward through layers — and see exactly where they vanish or explode.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # ── What Are Gradients? ──────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">📡 What Are Gradients?</h3>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                When a neural network makes a prediction, it compares its answer to the correct answer and calculates an <b>error</b>.
                Then it needs to figure out: <b>"How should I adjust each weight to reduce this error?"</b>
            </p>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                The answer is the <b>gradient</b> — a number that tells each weight which direction to move and how much.
                Gradients flow <b>backward</b> from the output layer to the input layer (this is called <b>backpropagation</b>).
            </p>
            <p style="font-size:0.9rem; margin-bottom:0;">
                The problem? At each layer, the gradient gets <b>multiplied</b> by the activation function's derivative.
                If that derivative is always < 1 (like Sigmoid's max of 0.25), the gradient shrinks exponentially — and early layers never learn!
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── The Telephone Game analogy ───────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem;">
            <h3 style="margin-top:0; color:var(--amber);">🎮 The Game of Telephone</h3>
            <p style="margin-bottom:0.75rem; font-size:0.9rem;">
                Imagine playing <b>telephone</b> with 10 people. You whisper "The cat sat on the mat" to person 1.
            </p>
            <p style="margin-bottom:0.75rem; font-size:0.9rem;">
                <b>Sigmoid network:</b> Each person only passes along 25% of what they heard.
                By person 10, the message is inaudible — "Th... ca... s..." — the first layers never get useful feedback.
            </p>
            <p style="margin-bottom:0; font-size:0.9rem;">
                <b>ReLU network:</b> Each person passes along 95% of what they heard.
                By person 10, the message is still clear — "The cat sat on the mat" — every layer learns effectively!
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Interactive Controls ─────────────────────────────────────
    st.markdown("<h2 style='margin-bottom:0.75rem;'>Simulate Gradient Flow</h2>", unsafe_allow_html=True)
    st.markdown("""
        <p style="margin-bottom:1rem; font-size:0.9rem;">
            Drag the slider to add more layers and watch what happens to the gradient at each layer.
            The <b>y-axis is logarithmic</b> — so each grid line represents a 10× difference!
        </p>
    """, unsafe_allow_html=True)

    depth = st.slider("Network Depth (Layers)", 2, 20, 10)

    grad_sigmoid = [1.0 * (0.25 ** i) for i in range(depth)]
    grad_relu    = [1.0 * (0.95 ** i) for i in range(depth)]
    grad_leaky   = [1.0 * (0.99 ** i) for i in range(depth)]
    layers_label = [f"L{i+1}" for i in range(depth)]

    def bar_color(val):
        if val < 1e-4: return '#e05d44'
        if val < 0.1:  return '#c8902a'
        return '#1a1714'

    # ── Three-way comparison ─────────────────────────────────────
    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown("""
            <div class="section-label">Sigmoid — Exponential Decay</div>
            <h3 style="margin-bottom:0.5rem;">Gradients Vanish ❌</h3>
        """, unsafe_allow_html=True)
        fig1 = go.Figure(go.Bar(
            x=layers_label, y=grad_sigmoid,
            marker_color=[bar_color(v) for v in grad_sigmoid]
        ))
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
            yaxis_type="log", yaxis_title="Gradient (log)",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            yaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            height=320, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown(f"""
            <div class="glass-card" style="border-left:3px solid #e05d44; padding:0.7rem 1rem;">
                <p style="margin:0; font-size:0.82rem;">
                    Layer {depth} gradient: <b>{grad_sigmoid[-1]:.2e}</b><br>
                    That's <b>{1/grad_sigmoid[-1]:.0f}×</b> weaker than the output!
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="section-label">Leaky ReLU — Gentle Decay</div>
            <h3 style="margin-bottom:0.5rem;">Gradients Survive ✅</h3>
        """, unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=layers_label, y=grad_leaky,
            marker_color=[bar_color(v) for v in grad_leaky]
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
            yaxis_type="log", yaxis_title="Gradient (log)",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            yaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            height=320, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"""
            <div class="glass-card" style="border-left:3px solid var(--amber); padding:0.7rem 1rem;">
                <p style="margin:0; font-size:0.82rem;">
                    Layer {depth} gradient: <b>{grad_leaky[-1]:.4f}</b><br>
                    Only <b>{(1-grad_leaky[-1])*100:.1f}%</b> loss — very manageable!
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="section-label">ReLU — Stable Flow</div>
            <h3 style="margin-bottom:0.5rem;">Gradients Thrive ✅</h3>
        """, unsafe_allow_html=True)
        fig3 = go.Figure(go.Bar(
            x=layers_label, y=grad_relu,
            marker_color=[bar_color(v) for v in grad_relu]
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
            yaxis_type="log",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            yaxis=dict(gridcolor='rgba(0,0,0,0.06)', color='chocolate', tickfont=dict(color='chocolate')),
            height=320, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown(f"""
            <div class="glass-card" style="border-left:3px solid #1a1714; padding:0.7rem 1rem;">
                <p style="margin:0; font-size:0.82rem;">
                    Layer {depth} gradient: <b>{grad_relu[-1]:.4f}</b><br>
                    Only <b>{(1-grad_relu[-1])*100:.1f}%</b> loss — excellent!
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── Warning/Success indicators ───────────────────────────────
    if grad_sigmoid[-1] < 1e-4:
        st.markdown("""
            <div class="glass-card" style="border-left:3px solid #e05d44; padding:0.9rem 1.2rem; margin-top:1rem; color:chocolate;">
                <span style="color: chocolate !important;">⚠️ <b>Vanishing Gradients Detected!</b> With Sigmoid at this depth, early layers receive 
                near-zero corrections. They essentially stop learning — this is why Sigmoid fell out of favor for hidden layers.</span>
            </div>
        """, unsafe_allow_html=True)

    # ── The Math Behind It ───────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <h2 style="margin-bottom:1rem;">The Math Behind It</h2>
    """, unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="large")
    with mc1:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top:0;">Chain Rule = Multiplication</h3>
                <p style="font-size:0.88rem; margin-bottom:0.5rem;">
                    Backpropagation uses the <b>chain rule</b> of calculus. For a network with n layers:
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial a_n} \cdot \prod_{i=1}^{n} f'_i(z_i)")
        st.markdown("""
            <p style="font-size:0.85rem;">
                Each layer <b>multiplies</b> the gradient by its derivative f'(z). If f'(z) is always small 
                (like Sigmoid's max of 0.25), the product shrinks exponentially.
            </p>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown("""
            <div class="glass-card">
                <h3 style="margin-top:0;">Why It Matters</h3>
                <p style="font-size:0.88rem;">
                    <b>Sigmoid:</b> max derivative = 0.25<br>
                    After 10 layers: 0.25¹⁰ = <b>0.0000009536</b> ← almost zero!
                </p>
                <p style="font-size:0.88rem; margin-bottom:0;">
                    <b>ReLU:</b> derivative = 1 (for positive inputs)<br>
                    After 10 layers: 1¹⁰ = <b>1.0</b> ← perfect gradient flow!
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── Key Takeaways ────────────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3, gap="medium")
    takeaways = [
        ("🚫", "Vanishing Gradients", "When derivatives are < 1 at every layer, gradients shrink exponentially. Sigmoid and Tanh are the worst offenders. Solution: use ReLU-family functions."),
        ("💥", "Exploding Gradients", "When derivatives are > 1 at every layer, gradients grow exponentially. Less common but equally deadly. Solution: gradient clipping and batch normalization."),
        ("✅", "Healthy Gradients", "The ideal: derivatives close to 1, so gradients flow smoothly through all layers. This is why ReLU (derivative = 1 for positive inputs) revolutionized deep learning."),
    ]
    for col, (icon, title, desc) in zip([tc1, tc2, tc3], takeaways):
        with col:
            st.markdown(f"""
                <div class="service-card animate-up" style="min-height:200px;">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                    <h3 style="margin:0 0 0.4rem 0; font-size:0.85rem;">{title}</h3>
                    <p style="font-size:0.8rem; margin:0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    # ── Try This ─────────────────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up" style="margin-top:1.5rem; border-left:4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🧪 Try This!</h3>
            <p style="font-size:0.88rem; margin-bottom:0;">
                Move the depth slider to <b>20 layers</b> and watch Sigmoid's gradient at Layer 20.
                It's approximately <b>0.25²⁰ ≈ 0.000000000001</b> — that's a trillion times weaker than the original signal!
                Meanwhile, ReLU's gradient at Layer 20 is still <b>0.36</b>. This is why modern deep networks 
                (ResNet has 152 layers!) use ReLU and its variants.
            </p>
        </div>
    """, unsafe_allow_html=True)
