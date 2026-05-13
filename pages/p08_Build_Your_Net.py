import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.ml_models import get_dataset, train_mlp, create_decision_boundary_data

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">08 / 10 &nbsp;·&nbsp; Hands-On</div>
            <h1>Build Your<br>Own Network</h1>
            <p style="max-width:500px;">Configure architecture, choose a dataset, then train a live model right in the browser.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # ── How Training Works ───────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🏗️ How Training Works</h3>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                Training a neural network is like teaching a student — you show it examples, it guesses, you tell it how wrong it was, 
                and it adjusts. This loop repeats thousands of times until it learns the pattern.
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem 2rem;">
                <p style="margin:0; font-size:0.85rem;"><b>1. Forward Pass</b> — Data flows through the network, producing a prediction.</p>
                <p style="margin:0; font-size:0.85rem;"><b>2. Loss Calculation</b> — Compare prediction to the correct answer.</p>
                <p style="margin:0; font-size:0.85rem;"><b>3. Backward Pass</b> — Gradients flow backward to figure out blame.</p>
                <p style="margin:0; font-size:0.85rem;"><b>4. Weight Update</b> — Each weight adjusts to reduce the error.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Dataset explanations ─────────────────────────────────────
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem;">
            <h3 style="margin-top:0; color:var(--amber);">📊 Understanding the Datasets</h3>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem 2rem;">
                <p style="margin:0; font-size:0.85rem;">🌙 <b>Moons</b> — Two interlocking crescents. Needs non-linear boundary. <i>Difficulty: Easy</i></p>
                <p style="margin:0; font-size:0.85rem;">⭕ <b>Circles</b> — One ring inside another. Needs a circular boundary. <i>Difficulty: Medium</i></p>
                <p style="margin:0; font-size:0.85rem;">📏 <b>Linear</b> — Linearly separable. Even a linear model can solve this. <i>Difficulty: Trivial</i></p>
                <p style="margin:0; font-size:0.85rem;">❌ <b>XOR</b> — The classic unsolvable-by-linear problem. <i>Difficulty: Hard</i></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.markdown("<h3 style='margin-bottom:1rem;'>Architecture</h3>", unsafe_allow_html=True)
        dataset_name  = st.selectbox("Dataset",             ["Moons","Circles","Linear","XOR"])
        activation    = st.selectbox("Activation Function", ["ReLU","Tanh","Sigmoid","Identity (Linear)"])
        hidden_layers = st.slider("Hidden Layers",      1, 4,    2)
        neurons       = st.slider("Neurons per Layer",  4, 64,   16)
        epochs        = st.slider("Max Epochs",        10, 1000, 200, 10)
        train_btn     = st.button("🚀 Train Model", key="train_btn")

        # ── Architecture visualization ───────────────────────────
        st.markdown("<div class='section-label' style='margin-top:1rem;'>Your Architecture</div>", unsafe_allow_html=True)
        arch_str = " → ".join(
            ["Input (2)"] +
            [f"Dense({neurons}, {activation})" for _ in range(hidden_layers)] +
            ["Output (1)"]
        )
        st.code(arch_str, language=None)

        total_params = 2 * neurons + neurons  # first layer
        for _ in range(hidden_layers - 1):
            total_params += neurons * neurons + neurons
        total_params += neurons + 1  # output layer
        st.markdown(f"""
            <div class="glass-card" style="padding:0.7rem 1rem; margin-top:0.5rem;">
                <p style="margin:0; font-size:0.82rem;">
                    📐 <b>Total Parameters:</b> {total_params:,}<br>
                    📚 <b>Training for:</b> up to {epochs} epochs<br>
                    ⚡ <b>Activation:</b> {activation}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ── Hints based on selection ─────────────────────────────
        if activation == "Identity (Linear)" and dataset_name in ["XOR", "Circles", "Moons"]:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid #e05d44; margin-top:0.75rem;">
                    <p style="margin:0; font-size:0.82rem;">
                        ⚠️ <b>Warning:</b> A linear activation can't solve non-linear patterns! 
                        This will likely fail on the selected dataset. Try it to see why activation functions matter!
                    </p>
                </div>
            """, unsafe_allow_html=True)

        if dataset_name == "XOR" and hidden_layers == 1 and neurons < 8:
            st.markdown("""
                <div class="glass-card" style="border-left:3px solid var(--amber); margin-top:0.75rem;">
                    <p style="margin:0; font-size:0.82rem;">
                        💡 <b>Hint:</b> XOR is tricky! You might need at least 2 hidden layers 
                        or more neurons per layer to solve it reliably.
                    </p>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='margin-bottom:0.75rem;'>Decision Boundary</h3>", unsafe_allow_html=True)

        X, y = get_dataset(dataset_name, 300)

        def scatter_fig(X, y, xx=None, yy=None, Z=None):
            fig = go.Figure()
            if xx is not None:
                fig.add_trace(go.Contour(
                    x=xx[0], y=yy[:,0], z=Z,
                    colorscale=[[0,'#f0ede8'],[1,'#c8902a']],
                    opacity=0.45, showscale=False
                ))
            fig.add_trace(go.Scatter(
                x=X[y==0][:,0], y=X[y==0][:,1], mode='markers',
                marker=dict(color='#1a1714', size=6, line=dict(color='chocolate',width=1)), name='Class 0'
            ))
            fig.add_trace(go.Scatter(
                x=X[y==1][:,0], y=X[y==1][:,1], mode='markers',
                marker=dict(color='#c8902a', size=6, line=dict(color='chocolate',width=1)), name='Class 1'
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
                xaxis=dict(gridcolor='rgba(0,0,0,0.06)'), yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
                height=440, margin=dict(l=10,r=10,t=10,b=10),
                legend=dict(font=dict(color='#1a1714'))
            )
            return fig

        if train_btn:
            with st.spinner(f"Training {hidden_layers}×{neurons} network with {activation}…"):
                arch  = tuple([neurons] * hidden_layers)
                model = train_mlp(X, y, activation=activation, hidden_layer_sizes=arch, max_iter=epochs)
                acc   = model.score(X, y)
                xx, yy, Z = create_decision_boundary_data(model, X)
            st.plotly_chart(scatter_fig(X, y, xx, yy, Z), use_container_width=True)

            # ── Results with interpretation ──────────────────────
            if acc >= 0.95:
                emoji, verdict = "🏆", "Excellent! The network solved the pattern nearly perfectly."
                border_color = "#1a1714"
            elif acc >= 0.80:
                emoji, verdict = "👍", "Good result, but there's room for improvement. Try more neurons or layers."
                border_color = "var(--amber)"
            elif acc >= 0.60:
                emoji, verdict = "😐", "Mediocre. The network is struggling. Consider a different activation or more capacity."
                border_color = "#c8902a"
            else:
                emoji, verdict = "❌", "Poor result. The model can't learn this pattern with current settings."
                border_color = "#e05d44"

            st.markdown(f"""
                <div class="glass-card" style="border-left:3px solid {border_color}; padding:0.9rem 1.2rem;">
                    {emoji} <b>Accuracy: {acc*100:.2f}%</b> — {verdict}
                </div>
            """, unsafe_allow_html=True)

            # ── Specific feedback ────────────────────────────────
            if acc < 0.6 and activation == "Identity (Linear)" and dataset_name != "Linear":
                st.markdown("""
                    <div class="glass-card" style="border-left:3px solid #e05d44; margin-top:0.75rem;">
                        <p style="margin:0; font-size:0.85rem;">
                            📝 <b>This is expected!</b> You used a linear activation on a non-linear dataset.
                            A linear model can only draw straight lines — it physically can't solve Moons, Circles, or XOR.
                            Switch to <b>ReLU</b> or <b>Tanh</b> to see the difference!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif acc > 0.95:
                st.markdown("""
                    <div class="glass-card" style="border-left:3px solid #1a1714; margin-top:0.75rem;">
                        <p style="margin:0; font-size:0.85rem;">
                            🎓 <b>Great job!</b> The decision boundary fits the data well. 
                            Notice how the colored regions curve and bend around the data points — 
                            that's the activation function at work, bending the decision boundary!
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.plotly_chart(scatter_fig(X, y), use_container_width=True)
            st.markdown("""
                <div class="glass-card" style="padding:0.9rem 1.2rem;">
                    👆 Configure architecture on the left, then click <b>Train Model</b> to see the decision boundary appear.
                </div>
            """, unsafe_allow_html=True)

    # ── Experiments to Try ───────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card animate-up" style="border-left:4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🧪 Experiments To Try</h3>
            <ol style="margin-bottom:0; font-size:0.88rem;">
                <li><b>Linear on XOR:</b> Use "Identity (Linear)" activation on XOR dataset — watch it fail. Then switch to ReLU and see it succeed. This proves why non-linearity matters!</li>
                <li><b>Depth vs Width:</b> Compare 1 layer × 64 neurons vs 4 layers × 16 neurons on Circles. Which works better?</li>
                <li><b>Underfitting:</b> Use 1 layer × 4 neurons on XOR with only 50 epochs. The network doesn't have enough capacity or training time.</li>
                <li><b>Activation showdown:</b> Train the same architecture with ReLU, Tanh, and Sigmoid on Moons. Compare the accuracy — which converges best?</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
