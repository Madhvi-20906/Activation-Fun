import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.ml_models import get_dataset, train_mlp, create_decision_boundary_data

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">04 / 10 &nbsp;·&nbsp; Theory</div>
            <h1>Why It<br>Matters</h1>
            <p style="max-width:500px;">Why can't we use only linear functions? This is the most important concept in deep learning.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # ── The Core Problem ─────────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🎯 The Big Question</h3>
            <p style="font-size:0.95rem; margin-bottom:0.5rem;">
                <b>"If I stack 100 linear layers, don't I get a super powerful model?"</b>
            </p>
            <p style="margin-bottom:0; font-size:0.9rem;">
                No! And here's the shocking reason: <b>100 linear layers = 1 linear layer</b>. 
                Mathematically, any composition of linear functions is still just one linear function:
                <code>f(g(x)) = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂) = W'x + b'</code>
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── Analogy Section ──────────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:2rem;">
            <h3 style="margin-top:0; color:var(--amber);">🧩 The Checkerboard Puzzle</h3>
            <p style="margin-bottom:0.75rem;">
                Imagine a checkerboard. With only a ruler (a linear function) you can draw one straight line —
                but no single straight line can separate all red squares from black squares.
            </p>
            <p style="margin-bottom:0.75rem;">
                Now imagine you could <b>fold the paper</b>. Suddenly, points that were far apart are now next to each other.
                That's exactly what activation functions do — they <b>warp and bend the input space</b> so that
                complex patterns become linearly separable!
            </p>
            <p style="margin-bottom:0;">
                This is why activation functions are <b>the most important ingredient</b> in neural networks.
                Without them, deep learning simply wouldn't work.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ── The XOR Problem ──────────────────────────────────────────
    st.markdown("""
        <h2 style="margin-bottom:0.5rem;">Live Proof: The XOR Problem</h2>
        <p style="max-width:500px; margin-bottom:1.5rem;">XOR is the simplest problem that a linear model <b>cannot solve</b>. Watch the difference activation makes.</p>
    """, unsafe_allow_html=True)

    X, y = get_dataset("XOR", n_samples=200)

    col1, col2 = st.columns(2, gap="large")

    def make_fig(xx, yy, Z, X, y, colorscale):
        fig = go.Figure()
        fig.add_trace(go.Contour(x=xx[0], y=yy[:,0], z=Z,
                                  colorscale=colorscale, opacity=0.45, showscale=False))
        fig.add_trace(go.Scatter(x=X[y==0][:,0], y=X[y==0][:,1], mode='markers',
                                  marker=dict(color='#1a1714', size=7, line=dict(color='chocolate', width=1)),
                                  name='Class 0'))
        fig.add_trace(go.Scatter(x=X[y==1][:,0], y=X[y==1][:,1], mode='markers',
                                  marker=dict(color='#c8902a', size=7, line=dict(color='chocolate', width=1)),
                                  name='Class 1'))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)'), yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
            height=380, margin=dict(l=10,r=10,t=10,b=10),
            legend=dict(font=dict(color='#1a1714'))
        )
        return fig

    with col1:
        st.markdown("""
            <div class="section-label">Without Activation — Linear</div>
            <h3 style="margin-bottom:0.5rem;">Only Straight Lines</h3>
        """, unsafe_allow_html=True)
        with st.spinner("Training linear model…"):
            model_lin = train_mlp(X, y, activation='Identity (Linear)', hidden_layer_sizes=(8,8), max_iter=300)
            acc_lin   = model_lin.score(X, y)
            xx, yy, Z_lin = create_decision_boundary_data(model_lin, X)
        st.plotly_chart(make_fig(xx, yy, Z_lin, X, y, 'Greys'), use_container_width=True)
        st.metric("Accuracy", f"{acc_lin*100:.1f}%")
        st.markdown("""
            <div class="glass-card" style="border-left:3px solid #e05d44;">
                <p style="margin:0; font-size:0.85rem;">
                    ❌ <b>Fails miserably!</b> The linear model can only draw a straight line. 
                    No matter how many linear layers you stack, it will never solve XOR — 
                    mathematically impossible.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="section-label">With Activation — ReLU</div>
            <h3 style="margin-bottom:0.5rem;">Curved Boundaries</h3>
        """, unsafe_allow_html=True)
        with st.spinner("Training ReLU model…"):
            model_relu = train_mlp(X, y, activation='ReLU', hidden_layer_sizes=(8,8), max_iter=300)
            acc_relu   = model_relu.score(X, y)
            xx, yy, Z_relu = create_decision_boundary_data(model_relu, X)
        st.plotly_chart(make_fig(xx, yy, Z_relu, X, y, [[0,'#f0ede8'],[1,'#c8902a']]), use_container_width=True)
        st.metric("Accuracy", f"{acc_relu*100:.1f}%")
        st.markdown("""
            <div class="glass-card" style="border-left:3px solid #1a1714;">
                <p style="margin:0; font-size:0.85rem;">
                    ✅ <b>Solves it easily!</b> ReLU lets the network bend and fold the decision boundary,
                    creating complex shapes that perfectly separate the two classes.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # ── Deeper Understanding ─────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <h2 style="margin-bottom:1rem;">Understanding Non-Linearity</h2>
    """, unsafe_allow_html=True)

    dc1, dc2, dc3 = st.columns(3, gap="medium")
    insights = [
        ("🏗️", "Representation Power", "What It Gives",
         "Non-linear activations let each layer learn a different 'transformation' of the data. Layer 1 might learn edges, Layer 2 learns shapes, Layer 3 learns objects."),
        ("📐", "Universal Approximation", "The Theorem",
         "A neural network with just ONE hidden layer and a non-linear activation can approximate ANY continuous function — given enough neurons. This is mathematically proven!"),
        ("🌊", "Space Warping", "The Intuition",
         "Each non-linear layer 'folds' the input space in a new way. After enough folds, even the most tangled data becomes separable by a simple straight line."),
    ]
    for col, (icon, title, label, desc) in zip([dc1, dc2, dc3], insights):
        with col:
            st.markdown(f"""
                <div class="service-card animate-up" style="min-height:220px;">
                    <div style="font-size:1.8rem; margin-bottom:0.5rem;">{icon}</div>
                    <div class="section-label" style="margin-bottom:0.2rem;">{label}</div>
                    <h3 style="margin:0 0 0.4rem 0; font-size:0.85rem;">{title}</h3>
                    <p style="font-size:0.8rem; margin:0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)

    # ── Real World Analogy ───────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card animate-up" style="border-left:4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🎨 Real-World Analogy: The Art Class</h3>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                Imagine you're in art class and the teacher asks you to draw a <b>circle</b>.
            </p>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                <b>Linear only:</b> You can only use a ruler. No matter how many rulers you use or how cleverly you combine them,
                you'll only ever draw straight lines. A circle is impossible.
            </p>
            <p style="font-size:0.9rem; margin-bottom:0;">
                <b>With activation:</b> Now you also have a <b>compass</b> (the activation function). 
                Suddenly you can draw curves! Combine enough curves and you can approximate ANY shape — 
                circles, spirals, even a portrait. That's the power of non-linearity.
            </p>
        </div>
    """, unsafe_allow_html=True)
