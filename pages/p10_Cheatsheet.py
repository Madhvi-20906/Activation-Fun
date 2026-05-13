import streamlit as st
import pandas as pd

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">10 / 10 &nbsp;·&nbsp; Reference</div>
            <h1>Cheatsheet</h1>
            <p style="max-width:500px;">Your quick-reference guide for interviews, implementation, and daily ML work.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # ── Decision Flowchart ───────────────────────────────────────
    st.markdown("""
        <h2 style="margin-bottom:0.75rem;">🧭 Which Activation Should I Use?</h2>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">Decision Flowchart</h3>
            <p style="font-size:0.9rem; margin-bottom:0.75rem;">
                Follow this decision tree to pick the right activation function in 30 seconds:
            </p>
            <p style="font-size:0.88rem; margin:0.3rem 0;">
                <b>Q: Is this for the output layer or a hidden layer?</b>
            </p>
            <div style="padding-left:1.5rem; border-left:2px solid var(--stone-300); margin:0.5rem 0;">
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    📤 <b>Output layer — binary classification?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">Sigmoid</span>
                </p>
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    📤 <b>Output layer — multi-class classification?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">Softmax</span>
                </p>
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    📤 <b>Output layer — regression?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">No activation (Linear)</span>
                </p>
            </div>
            <div style="padding-left:1.5rem; border-left:2px solid var(--amber); margin:0.5rem 0;">
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    🔧 <b>Hidden layer — Transformer/LLM?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">GELU</span>
                </p>
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    🔧 <b>Hidden layer — RNN/LSTM?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">Tanh</span>
                </p>
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    🔧 <b>Hidden layer — CNN or general?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">ReLU</span> (or Leaky ReLU if worried about dead neurons)
                </p>
                <p style="font-size:0.85rem; margin:0.2rem 0;">
                    🔧 <b>Hidden layer — very deep network (100+ layers)?</b> → <span style="background:var(--stone-200); padding:2px 8px; border-radius:4px; font-weight:700;">ReLU + Residual Connections</span>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Comparison Table ─────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:0.75rem;'>Quick Comparison</h3>", unsafe_allow_html=True)

    data = {
        "Function":    ["ReLU","Sigmoid","Tanh","Softmax","Leaky ReLU","ELU","GELU","Swish"],
        "Formula":     ["max(0, x)","1/(1+e⁻ˣ)","tanh(x)","eˣⁱ/Σeˣʲ","max(0.01x, x)","x if x>0 else α(eˣ-1)","x·Φ(x)","x·σ(x)"],
        "Range":       ["[0, ∞)","(0, 1)","(−1, 1)","(0,1) per class","(−∞, ∞)","(−α, ∞)","(−0.17, ∞)","(−0.28, ∞)"],
        "Best For":    ["CNN hidden layers","Binary output","RNN/LSTM state","Multi-class output","Safe deep nets","Smooth deep nets","Transformers / LLMs","EfficientNet / deep nets"],
        "Watch Out":   ["Dead neurons","Vanishing gradients","Vanishing gradients","Only output layer!","Leak coefficient","Slower than ReLU","Compute cost","Compute cost"],
        "Speed":       ["⚡⚡⚡","⚡","⚡","⚡⚡","⚡⚡⚡","⚡⚡","⚡","⚡"],
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    # ── Key Formulas ─────────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:0.75rem;'>Key Formulas</h3>", unsafe_allow_html=True)

    fc1, fc2 = st.columns(2, gap="large")
    with fc1:
        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">ReLU</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x) = \max(0, x), \quad f'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \le 0 \end{cases}")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">Sigmoid</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"\sigma(x) = \frac{1}{1+e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">Tanh</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">Softmax</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}")

    with fc2:
        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">Leaky ReLU</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x) = \max(\alpha x, x), \quad f'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \le 0 \end{cases}")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">ELU</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">GELU</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x) = x \cdot \Phi(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)")

        st.markdown("""
            <div class="glass-card" style="margin-bottom:0.75rem;">
                <div class="section-label" style="margin-bottom:0.2rem;">Swish</div>
            </div>
        """, unsafe_allow_html=True)
        st.latex(r"f(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}")

    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    # ── Interview Q&A ────────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:0.75rem;'>Interview Q&A</h3>", unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
            <p style="margin:0; font-size:0.88rem;">
                📋 These are the <b>most commonly asked</b> activation function questions in ML interviews.
                Click each one to reveal a complete, interview-ready answer.
            </p>
        </div>
    """, unsafe_allow_html=True)

    qa = [
        ("Why can't we use only linear activations?",
         "Without non-linear activations, a neural network — no matter how deep — behaves exactly like a single-layer linear regression. Mathematically, composing linear functions gives another linear function: f(g(x)) = W₂(W₁x + b₁) + b₂ = W'x + b'. It cannot learn complex patterns like XOR, image features, or language."),
        ("What is the dying ReLU problem?",
         "If a neuron using ReLU consistently receives large negative inputs (often due to a large negative bias or unlucky weight initialization), its output is always 0 and its gradient is always 0. It stops updating entirely and 'dies.' Solutions: use Leaky ReLU, He initialization, or careful learning rate tuning."),
        ("Why does ChatGPT use GELU instead of ReLU?",
         "GELU weights each input by its probability under a Gaussian distribution (Φ(x)). This creates a smooth, probabilistic gate rather than ReLU's hard cutoff at zero. For Transformers, this smoother activation creates richer representations in the feed-forward blocks, leading to better language understanding. It was adopted by BERT, then GPT-2, and is now standard."),
        ("When should I use Softmax?",
         "Only in the output layer of a multi-class classification network. It converts raw logits into a probability distribution that sums to 1.0. Never use it in hidden layers — it forces all neuron outputs to compete (sum to 1), killing gradient diversity and independent feature learning."),
        ("What's the difference between Sigmoid and Tanh?",
         "Tanh is a rescaled Sigmoid: tanh(x) = 2σ(2x) − 1. The key difference: Sigmoid outputs (0,1) while Tanh outputs (-1,1). Tanh is zero-centered, meaning gradients don't all push in the same direction — this avoids the zig-zag optimization problem and often converges faster."),
        ("Why did ReLU replace Sigmoid in hidden layers?",
         "Three reasons: (1) ReLU's gradient is 1 for positive inputs — no vanishing gradient, enabling deeper networks. (2) ReLU is computationally trivial — just max(0,x), no exponentials. (3) ReLU creates sparse activations — many neurons output 0, which acts as implicit regularization. AlexNet (2012) showed ReLU trains 6× faster than Tanh."),
        ("What is the 'temperature' in Softmax?",
         "Temperature T divides the logits before Softmax: softmax(x/T). T<1 makes the distribution sharper (more confident), T>1 makes it flatter (more random). T→0 gives argmax (pick the best), T→∞ gives uniform distribution. ChatGPT uses temperature to control creativity."),
        ("How does Swish differ from ReLU?",
         "Swish(x) = x·σ(x) is smooth everywhere (unlike ReLU's sharp corner at 0) and is non-monotonic (it dips slightly below zero near x≈−1.28). This non-monotonicity gives extra representational power. Google discovered it via automated search and found it consistently outperforms ReLU in deep networks like EfficientNet."),
    ]

    for q, a in qa:
        with st.expander(q):
            st.write(a)

    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

    # ── Common Mistakes ──────────────────────────────────────────
    st.markdown("<h3 style='margin-bottom:0.75rem;'>Common Mistakes to Avoid</h3>", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2, gap="medium")
    mistakes = [
        ("❌", "Using Sigmoid in hidden layers", "Causes vanishing gradients in deep networks. Use ReLU instead."),
        ("❌", "Using Softmax in hidden layers", "Forces all outputs to sum to 1, killing feature diversity."),
        ("❌", "Using ReLU for output layer", "Outputs are unbounded and can't represent probabilities."),
        ("❌", "Ignoring dead neurons", "Monitor ReLU activations — switch to Leaky ReLU if many are zero."),
        ("❌", "Not matching activation to task", "Binary → Sigmoid, Multi-class → Softmax, Regression → Linear."),
        ("❌", "Using the same activation everywhere", "Modern networks mix: GELU in hidden, Softmax in output, etc."),
    ]
    for i, (icon, title, desc) in enumerate(mistakes):
        with [mc1, mc2][i % 2]:
            st.markdown(f"""
                <div class="glass-card" style="margin-bottom:0.75rem; border-left:3px solid #e05d44;">
                    <p style="margin:0; font-size:0.85rem;">{icon} <b>{title}</b><br>{desc}</p>
                </div>
            """, unsafe_allow_html=True)
