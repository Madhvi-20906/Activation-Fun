import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.math_helpers import relu

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">02 / 10 &nbsp;·&nbsp; Fundamentals</div>
            <h1>Neural Network<br>Basics</h1>
            <p style="max-width:500px;">Learn how a single neuron processes information — inputs, weights, bias, and activation.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # ── Step-by-step explanation ──────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🧬 What Is a Neuron?</h3>
            <p style="margin-bottom:0.75rem; font-size:0.9rem;">
                A <b>neuron</b> is the smallest building block of a neural network. It mimics how brain cells work:
                it receives signals, processes them, and decides whether to "fire" an output.
            </p>
            <p style="margin-bottom:0; font-size:0.9rem;">
                Every neuron performs exactly <b>3 steps</b>:
            </p>
        </div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3, gap="medium")
    with sc1:
        st.markdown("""
            <div class="service-card animate-up delay-1" style="text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">⚡</div>
                <h3 style="margin:0 0 0.3rem 0; font-size:0.85rem;">Step 1: Weighted Sum</h3>
                <p style="font-size:0.82rem; margin:0;">Multiply each input by its weight, then add them all together with a bias term.</p>
            </div>
        """, unsafe_allow_html=True)
    with sc2:
        st.markdown("""
            <div class="service-card animate-up delay-2" style="text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">🎛️</div>
                <h3 style="margin:0 0 0.3rem 0; font-size:0.85rem;">Step 2: Activation</h3>
                <p style="font-size:0.82rem; margin:0;">Pass the sum through an activation function — this introduces non-linearity (the "magic").</p>
            </div>
        """, unsafe_allow_html=True)
    with sc3:
        st.markdown("""
            <div class="service-card animate-up delay-3" style="text-align:center;">
                <div style="font-size:2rem; margin-bottom:0.5rem;">📤</div>
                <h3 style="margin:0 0 0.3rem 0; font-size:0.85rem;">Step 3: Output</h3>
                <p style="font-size:0.82rem; margin:0;">The activated value becomes the neuron's output — which may feed into the next layer.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    # ── Interactive Neuron ────────────────────────────────────────
    st.markdown("""<h2 style="margin-bottom:0.5rem;">Interactive Neuron</h2>
        <p style="max-width:480px; margin-bottom:1.5rem;">Drag the sliders to change inputs, weights, and bias — watch the neuron compute its output in real-time.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""
            <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem;">
                <h3 style="margin-top:0;">Adjust Parameters</h3>
        """, unsafe_allow_html=True)
        x1 = st.slider("Input 1 (x₁)", -5.0, 5.0, 1.0, 0.1)
        x2 = st.slider("Input 2 (x₂)", -5.0, 5.0, -2.0, 0.1)
        w1 = st.slider("Weight 1 (w₁)", -5.0, 5.0, 2.0, 0.1)
        w2 = st.slider("Weight 2 (w₂)", -5.0, 5.0, 1.0, 0.1)
        b  = st.slider("Bias (b)", -5.0, 5.0, 0.0, 0.1)
        st.markdown("</div>", unsafe_allow_html=True)

        z = w1*x1 + w2*x2 + b
        a = relu(z)

        st.markdown("""<h3 style="margin-bottom:0.5rem;">The Math</h3>""", unsafe_allow_html=True)
        st.latex(r"z = (w_1 \cdot x_1) + (w_2 \cdot x_2) + b")
        st.latex(f"z = ({w1:.1f} \\cdot {x1:.1f}) + ({w2:.1f} \\cdot {x2:.1f}) + {b:.1f} = {z:.2f}")
        st.latex(r"a = \text{ReLU}(z) = \max(0, z)")
        st.latex(f"a = {a:.2f}")

        # Dynamic insight based on current values
        if z < 0:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid #e05d44; margin-top:1rem;">
                    <p style="margin:0; font-size:0.85rem;">🔴 <b>Neuron is OFF!</b> The weighted sum is negative, so ReLU outputs 0. 
                    This neuron won't pass any signal forward. Try increasing the inputs or weights to "wake it up."</p>
                </div>
            """, unsafe_allow_html=True)
        elif z > 0 and z < 1:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid var(--amber); margin-top:1rem;">
                    <p style="margin:0; font-size:0.85rem;">🟡 <b>Weak signal.</b> The neuron is barely active. 
                    In a real network, this neuron would contribute only a small amount to the next layer.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid #1a1714; margin-top:1rem;">
                    <p style="margin:0; font-size:0.85rem;">🟢 <b>Strong signal! (a = {a:.2f})</b> The neuron is firing with high confidence. 
                    This strong activation will significantly influence the next layer's computations.</p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""<h3 style="margin-bottom:0.75rem;">Visual Flow</h3>""", unsafe_allow_html=True)
        fig = go.Figure()

        node_x = [0, 0, 1, 2]
        node_y = [1, -1, 0, 0]
        node_text = [f"x₁={x1}", f"x₂={x2}", f"Σ={z:.2f}", f"a={a:.2f}"]
        node_colors = ['#cec7bc', '#cec7bc', '#a8a094', '#1a1714' if a > 0 else '#e3ddd5']

        fig.add_trace(go.Scatter(x=[0,1], y=[1,0], mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=max(1,abs(w1))), showlegend=False))
        fig.add_trace(go.Scatter(x=[0,1], y=[-1,0], mode='lines', line=dict(color='rgba(0,0,0,0.2)', width=max(1,abs(w2))), showlegend=False))
        fig.add_trace(go.Scatter(x=[1,2], y=[0,0], mode='lines', line=dict(color='#c8902a' if a > 0 else 'rgba(0,0,0,0.1)', width=4), showlegend=False))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_text, textposition="top center",
            textfont=dict(color='#1a1714', size=12, family="Syne"),
            marker=dict(size=44, color=node_colors, line=dict(color='rgba(0,0,0,0.15)', width=2)),
            showlegend=False
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=380, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="glass-card animate-up delay-2">
            <h3 style="margin-top:0; color:var(--amber);">🧠 Analogy: The Umbrella Decision</h3>
            <ul style="margin-bottom:0;">
                <li><b>Inputs (x):</b> The clues you gather — is it cloudy? What does the weather app say?</li>
                <li><b>Weights (w):</b> How much you trust each clue. You trust the weather app (high w) more than your friend's hunch (low w).</li>
                <li><b>Bias (b):</b> Your personal tendency — always cautious (+b) or always optimistic (−b)?</li>
                <li><b>Activation:</b> The final decision — grab the umbrella, or leave it? If the combined evidence is negative, you don't act (ReLU = 0).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card animate-up delay-3" style="margin-top: 1rem;">
            <h3 style="margin-top:0; color:var(--stone-900);">💡 Did you know?</h3>
            <p style="margin-bottom:0; font-size:0.9rem;">
                Biological neurons in our brain work somewhat similarly! They collect electrical signals from dendrites (inputs & weights), build up a charge (summation), and if the charge crosses a certain threshold (bias & activation), the neuron "fires" a spike of electricity down its axon. Artificial neural networks were inspired by this very mechanism!
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Key Concepts Summary ──────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="animate-up">
            <h2 style="margin-bottom:1rem;">Key Concepts</h2>
        </div>
    """, unsafe_allow_html=True)

    kc1, kc2, kc3, kc4 = st.columns(4, gap="medium")
    concepts = [
        ("🔢", "Weights", "Learnable parameters that control how much importance each input gets. The network adjusts these during training."),
        ("➕", "Bias", "A constant added to shift the activation. Think of it as the neuron's default 'mood' before seeing any input."),
        ("⚡", "Activation", "A non-linear function applied after summation. Without it, the entire network would be just one big linear equation."),
        ("🔄", "Backpropagation", "The learning algorithm — it calculates how wrong the output was, then sends corrections backward through every layer."),
    ]
    for col, (icon, title, desc) in zip([kc1, kc2, kc3, kc4], concepts):
        with col:
            st.markdown(f"""
                <div class="service-card animate-up" style="min-height:200px;">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                    <h3 style="margin:0 0 0.4rem 0; font-size:0.85rem;">{title}</h3>
                    <p style="font-size:0.8rem; margin:0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    # ── Try This experiment ───────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card animate-up" style="border-left:4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🧪 Try This Experiment!</h3>
            <p style="font-size:0.88rem; margin-bottom:0.5rem;">Go back to the sliders above and try these challenges:</p>
            <ol style="margin-bottom:0; font-size:0.85rem;">
                <li><b>Kill the neuron:</b> Set both inputs to negative values. Watch the output drop to 0 — this is the "dead neuron" problem!</li>
                <li><b>Flip the sign:</b> Set w₁ = −3 and x₁ = 2. Now the weight says "this input matters a lot, but in the OPPOSITE direction."</li>
                <li><b>Bias power:</b> Set all inputs and weights to 0, then increase only the bias. The bias alone can activate the neuron!</li>
                <li><b>Weight battle:</b> Set x₁ = 5, w₁ = 2, x₂ = 5, w₂ = −2. The weights cancel out — the neuron is confused!</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
