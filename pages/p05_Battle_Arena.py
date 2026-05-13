import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.math_helpers import ACTIVATION_FUNCTIONS

# ── Verdict data for every pair ──────────────────────────────────
VERDICTS = {
    "ReLU": {
        "strength": "Speed & simplicity",
        "weakness": "Dead neurons",
        "best_for": "CNNs, hidden layers",
    },
    "Sigmoid": {
        "strength": "Probabilistic output",
        "weakness": "Vanishing gradients",
        "best_for": "Binary output layer",
    },
    "Tanh": {
        "strength": "Zero-centered",
        "weakness": "Vanishing gradients",
        "best_for": "RNN/LSTM states",
    },
    "Leaky ReLU": {
        "strength": "No dead neurons",
        "weakness": "Leak value is arbitrary",
        "best_for": "Safe ReLU replacement",
    },
    "ELU": {
        "strength": "Smooth gradients",
        "weakness": "Slower computation",
        "best_for": "Deep nets, smooth optimization",
    },
    "GELU": {
        "strength": "Probabilistic gating",
        "weakness": "Compute-heavy",
        "best_for": "Transformers, LLMs",
    },
    "Swish": {
        "strength": "Self-gating, non-monotonic",
        "weakness": "Compute-heavy",
        "best_for": "EfficientNet, deep nets",
    },
    "Softmax": {
        "strength": "Perfect probability distribution",
        "weakness": "Only for output layer",
        "best_for": "Multi-class classification",
    },
}

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">05 / 10 &nbsp;·&nbsp; Compare</div>
            <h1>Battle<br>Arena</h1>
            <p style="max-width:500px;">Head-to-head comparison of any two activation functions — curves, gradients, radar stats, and a final verdict.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">⚔️ How Battles Work</h3>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                Choose two activation functions below and compare them on <b>5 key dimensions</b>.
                The radar chart shows their strengths at a glance — the bigger the shape, the better overall.
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem 2rem;">
                <p style="margin:0; font-size:0.85rem;"><b>Training Speed</b> — How fast the network converges to a good solution.</p>
                <p style="margin:0; font-size:0.85rem;"><b>Gradient Stability</b> — How smoothly gradients flow backward (no vanishing/exploding).</p>
                <p style="margin:0; font-size:0.85rem;"><b>Compute Cost</b> — Higher = cheaper to run. ReLU is king here.</p>
                <p style="margin:0; font-size:0.85rem;"><b>Dead Neuron Safety</b> — Higher = neurons can always recover and keep learning.</p>
                <p style="margin:0; font-size:0.85rem;"><b>Saturation Safety</b> — Higher = output doesn't flatten at extreme inputs.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        func_a = st.selectbox("🥊 Fighter A", list(ACTIVATION_FUNCTIONS.keys()), index=0)
    with col2:
        func_b = st.selectbox("🥊 Fighter B", list(ACTIVATION_FUNCTIONS.keys()), index=2)

    x = np.linspace(-5, 5, 200)

    if func_a == "Softmax" or func_b == "Softmax":
        st.info("Softmax is multi-dimensional — plotting a 1D approximation for visual comparison.")

    y_a = ACTIVATION_FUNCTIONS[func_a]["func"](x) if func_a != "Softmax" else np.exp(x)/np.sum(np.exp(x))
    y_b = ACTIVATION_FUNCTIONS[func_b]["func"](x) if func_b != "Softmax" else np.exp(x)/np.sum(np.exp(x))

    # ── Curve Comparison ─────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:0.5rem;'>Curve Comparison</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
            <p style="margin:0; font-size:0.85rem;">
                📈 The <b>solid lines</b> show how each function transforms inputs.
                Notice where they differ most — that's where your choice of activation function matters!
            </p>
        </div>
    """, unsafe_allow_html=True)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=x, y=y_a, mode='lines', name=func_a,
                                   line=dict(color='#1a1714', width=2.5)))
    fig_line.add_trace(go.Scatter(x=x, y=y_b, mode='lines', name=func_b,
                                   line=dict(color='#c8902a', width=2.5)))
    fig_line.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.1)")
    fig_line.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.1)")
    fig_line.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
        xaxis=dict(gridcolor='rgba(0,0,0,0.06)'), yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
        height=340, margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(font=dict(color='#1a1714'))
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Gradient Comparison ──────────────────────────────────────
    if func_a != "Softmax" and func_b != "Softmax":
        st.markdown("<h3 style='margin-bottom:0.5rem;'>Gradient Comparison</h3>", unsafe_allow_html=True)
        st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
                <p style="margin:0; font-size:0.85rem;">
                    📉 The gradient (derivative) determines how well a network learns.
                    <b>Flat regions = vanishing gradients = slow/no learning.</b>
                    Compare where each function's gradient is strong vs weak.
                </p>
            </div>
        """, unsafe_allow_html=True)

        dy_a = ACTIVATION_FUNCTIONS[func_a]["deriv"](x)
        dy_b = ACTIVATION_FUNCTIONS[func_b]["deriv"](x)

        fig_grad = go.Figure()
        fig_grad.add_trace(go.Scatter(x=x, y=dy_a, mode='lines', name=f"{func_a} gradient",
                                       line=dict(color='#1a1714', width=2, dash='dash')))
        fig_grad.add_trace(go.Scatter(x=x, y=dy_b, mode='lines', name=f"{func_b} gradient",
                                       line=dict(color='#c8902a', width=2, dash='dash')))
        fig_grad.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.1)")
        fig_grad.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.1)")
        fig_grad.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)'), yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
            height=300, margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(font=dict(color='#1a1714'))
        )
        st.plotly_chart(fig_grad, use_container_width=True)

    # ── Radar Stats ──────────────────────────────────────────────
    stats = {
        "Sigmoid":    [2, 2, 8, 10, 2],
        "Tanh":       [3, 3, 7, 10, 2],
        "ReLU":       [9, 8, 10, 2, 10],
        "Leaky ReLU": [9, 8, 9, 8, 10],
        "ELU":        [8, 9, 7, 9, 10],
        "GELU":       [8, 10, 6, 9, 10],
        "Swish":      [8, 10, 5, 9, 10],
        "Softmax":    [5, 5, 6, 10, 2],
    }
    cats = ['Training Speed','Gradient Stability','Compute Cost','Dead Neuron Safety','Saturation Safety']

    st.markdown("<h3 style='margin-bottom:0.5rem;'>Radar Stats</h3>", unsafe_allow_html=True)
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=stats[func_a], theta=cats, fill='toself', name=func_a,
                                         line=dict(color='#1a1714'), fillcolor='rgba(26,23,20,0.12)'))
    fig_radar.add_trace(go.Scatterpolar(r=stats[func_b], theta=cats, fill='toself', name=func_b,
                                         line=dict(color='#c8902a'), fillcolor='rgba(200,144,42,0.15)'))
    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor='rgba(255,255,255,0.4)',
            radialaxis=dict(visible=True, range=[0,10], color='rgba(0,0,0,0.3)', gridcolor='rgba(0,0,0,0.08)'),
            angularaxis=dict(color='#1a1714', gridcolor='rgba(0,0,0,0.08)')
        ),
        height=420, margin=dict(l=40,r=40,t=40,b=40),
        legend=dict(font=dict(color='#1a1714'))
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Verdict Section ──────────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("<h2 style='margin-bottom:1rem;'>⚖️ Final Verdict</h2>", unsafe_allow_html=True)

    score_a = sum(stats[func_a])
    score_b = sum(stats[func_b])

    vc1, vc2 = st.columns(2, gap="large")
    with vc1:
        va = VERDICTS[func_a]
        st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid #1a1714;">
                <h3 style="margin-top:0;">{func_a}</h3>
                <div style="font-size:2rem; font-weight:800; font-family:'Syne'; color:#1a1714; margin-bottom:0.5rem;">{score_a}/50</div>
                <p style="font-size:0.85rem; margin:0.3rem 0;">💪 <b>Strength:</b> {va['strength']}</p>
                <p style="font-size:0.85rem; margin:0.3rem 0;">⚠️ <b>Weakness:</b> {va['weakness']}</p>
                <p style="font-size:0.85rem; margin:0.3rem 0;">🎯 <b>Best For:</b> {va['best_for']}</p>
            </div>
        """, unsafe_allow_html=True)
    with vc2:
        vb = VERDICTS[func_b]
        st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid #c8902a;">
                <h3 style="margin-top:0;">{func_b}</h3>
                <div style="font-size:2rem; font-weight:800; font-family:'Syne'; color:#c8902a; margin-bottom:0.5rem;">{score_b}/50</div>
                <p style="font-size:0.85rem; margin:0.3rem 0;">💪 <b>Strength:</b> {vb['strength']}</p>
                <p style="font-size:0.85rem; margin:0.3rem 0;">⚠️ <b>Weakness:</b> {vb['weakness']}</p>
                <p style="font-size:0.85rem; margin:0.3rem 0;">🎯 <b>Best For:</b> {vb['best_for']}</p>
            </div>
        """, unsafe_allow_html=True)

    # ── Winner announcement ──────────────────────────────────────
    if func_a == func_b:
        verdict_msg = f"🤝 <b>It's the same function!</b> Try picking two different ones to see a real battle."
    elif score_a > score_b:
        verdict_msg = f"🏆 <b>{func_a} wins on paper!</b> But remember — the best activation depends on your specific use case, not just raw scores. {func_b} might still be better for {vb['best_for']}."
    elif score_b > score_a:
        verdict_msg = f"🏆 <b>{func_b} wins on paper!</b> But remember — the best activation depends on your specific use case, not just raw scores. {func_a} might still be better for {va['best_for']}."
    else:
        verdict_msg = f"🤝 <b>It's a tie!</b> Both score {score_a}/50. Choose based on your use case: {func_a} excels at {va['best_for']}, while {func_b} excels at {vb['best_for']}."

    st.markdown(f"""
        <div class="glass-card animate-up" style="border-left:4px solid var(--amber); margin-top:1rem;">
            <p style="margin:0; font-size:0.9rem;">{verdict_msg}</p>
        </div>
    """, unsafe_allow_html=True)

    # ── Pro Tip ──────────────────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up" style="margin-top:1rem;">
            <h3 style="margin-top:0; color:var(--amber);">💡 Pro Tip: The Decision Shortcut</h3>
            <p style="font-size:0.88rem; margin-bottom:0;">
                Not sure which to pick? Use this <b>3-second rule</b>:<br>
                • <b>Building a CNN?</b> → Start with ReLU<br>
                • <b>Building a Transformer/LLM?</b> → Use GELU<br>
                • <b>Binary yes/no output?</b> → Sigmoid for the last layer<br>
                • <b>Multi-class output?</b> → Softmax for the last layer<br>
                • <b>RNN/LSTM?</b> → Tanh for hidden states<br>
                • <b>Worried about dead neurons?</b> → Swap ReLU for Leaky ReLU
            </p>
        </div>
    """, unsafe_allow_html=True)
