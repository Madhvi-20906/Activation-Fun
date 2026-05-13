import streamlit as st
import plotly.graph_objects as go
import numpy as np
from utils.math_helpers import ACTIVATION_FUNCTIONS

# ── Rich explanations for each function ──────────────────────────
EXPLANATIONS = {
    "ReLU": {
        "formula": r"f(x) = \max(0, x)",
        "desc": "A strict gate. Negative values are blocked (set to zero); positive values pass through unchanged.",
        "how_it_works": """
            Imagine a **bouncer at a club**. If you're on the guest list (positive value), you walk right in unchanged.
            If you're not on the list (negative value), you're turned away — output is zero.
            
            This simplicity is why ReLU is so fast: it's just a comparison operation, no exponentials needed!
        """,
        "when_to_use": "**Default choice** for hidden layers in most networks, especially CNNs (image recognition, object detection).",
        "when_not_to_use": "If many neurons receive consistently negative inputs — they'll permanently output 0 ('die') and never recover.",
        "pros": ["Extremely fast to compute", "No vanishing gradient (for positive inputs)", "Sparse activation (efficiency)"],
        "cons": ["Dead neurons (can't recover once stuck at 0)", "Not zero-centered", "Unbounded output can cause instability"],
        "fun_fact": "ReLU was first introduced in 2000, but it didn't become popular until 2012 when AlexNet used it to win ImageNet — proving it trains **6× faster** than Tanh!",
        "code_np": "np.maximum(0, x)",
        "code_tf": "tf.keras.activations.relu(x)",
        "code_pt": "torch.nn.ReLU()(x)",
    },
    "Sigmoid": {
        "formula": r"f(x) = \frac{1}{1 + e^{-x}}",
        "desc": "The probability machine — squashes any value into (0, 1). Great for binary output layers.",
        "how_it_works": """
            Think of a **dimmer switch** that smoothly goes from OFF (0) to ON (1).
            No matter how extreme the input, the output is always a gentle number between 0 and 1.
            
            This makes it perfect for answering **yes/no questions**: "Is this email spam?" → 0.92 means "92% sure it's spam."
        """,
        "when_to_use": "**Binary classification output layers** (spam detection, medical diagnosis, fraud detection).",
        "when_not_to_use": "Hidden layers of deep networks — gradients shrink exponentially and early layers stop learning.",
        "pros": ["Clear probabilistic interpretation", "Smooth gradient everywhere", "Output bounded to (0,1)"],
        "cons": ["Vanishing gradients at extremes", "Not zero-centered (slows convergence)", "Computationally expensive (exp)"],
        "fun_fact": "Sigmoid was the dominant activation function from the 1980s until ~2012. It's modeled after the firing rate of biological neurons!",
        "code_np": "1 / (1 + np.exp(-x))",
        "code_tf": "tf.keras.activations.sigmoid(x)",
        "code_pt": "torch.nn.Sigmoid()(x)",
    },
    "Tanh": {
        "formula": r"f(x) = \tanh(x)",
        "desc": "Like Sigmoid but zero-centered, squashing values into (−1, 1). Ideal for RNNs.",
        "how_it_works": """
            Imagine a **volume knob** that goes from −1 (full reverse) through 0 (silent) to +1 (full forward).
            
            Unlike Sigmoid, Tanh is **zero-centered** — meaning its average output is around 0.
            This helps the next layer because it receives both positive and negative signals, making learning faster.
        """,
        "when_to_use": "**RNN/LSTM hidden states** where you need values centered around zero. Also good for data that has negative values.",
        "when_not_to_use": "Very deep networks — still suffers from vanishing gradients (just not as badly as Sigmoid).",
        "pros": ["Zero-centered (faster convergence)", "Stronger gradients than Sigmoid", "Output bounded to (-1,1)"],
        "cons": ["Still has vanishing gradient at extremes", "Computationally more expensive than ReLU", "Saturates for large inputs"],
        "fun_fact": "Tanh is actually just a rescaled Sigmoid! The formula is: tanh(x) = 2·sigmoid(2x) − 1. They're mathematical siblings!",
        "code_np": "np.tanh(x)",
        "code_tf": "tf.keras.activations.tanh(x)",
        "code_pt": "torch.nn.Tanh()(x)",
    },
    "Leaky ReLU": {
        "formula": r"f(x) = \max(0.01x, x)",
        "desc": "A forgiving gate — lets a tiny fraction of negative values through to prevent dead neurons.",
        "how_it_works": """
            Remember the **bouncer**? Leaky ReLU is a *nicer* bouncer. Even if you're not on the list,
            they let you peek through the door (1% of your value passes through).
            
            This tiny leak means that even negative inputs get a small gradient — so neurons can always recover and keep learning.
        """,
        "when_to_use": "**Drop-in replacement for ReLU** whenever you're worried about dead neurons, especially in deep networks.",
        "when_not_to_use": "Rarely a bad choice! The 0.01 slope is a hyperparameter you can tune (called Parametric ReLU / PReLU).",
        "pros": ["Prevents dead neurons", "Nearly as fast as ReLU", "Small gradients for all inputs"],
        "cons": ["The leak coefficient (0.01) is arbitrary", "Doesn't always outperform ReLU in practice", "Results can be inconsistent"],
        "fun_fact": "Parametric ReLU (PReLU) lets the network LEARN the optimal leak coefficient! The paper showed it improved ImageNet accuracy with zero extra cost.",
        "code_np": "np.where(x > 0, x, 0.01 * x)",
        "code_tf": "tf.keras.layers.LeakyReLU(alpha=0.01)(x)",
        "code_pt": "torch.nn.LeakyReLU(0.01)(x)",
    },
    "ELU": {
        "formula": r"f(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \le 0 \end{cases}",
        "desc": "Smoother than Leaky ReLU — handles negatives with a gentle curve instead of a sharp angle.",
        "how_it_works": """
            ELU is like a **luxury version of Leaky ReLU**. Instead of a sharp kink at zero, 
            it uses a smooth exponential curve for negative values.
            
            This smoothness means the gradient changes gradually (no sudden jumps), 
            which helps optimization algorithms find better solutions faster.
        """,
        "when_to_use": "**Deep networks where you want smooth gradients** — often gives faster convergence than ReLU or Leaky ReLU.",
        "when_not_to_use": "When speed is critical — the exponential computation for negatives is slower than simple ReLU.",
        "pros": ["Smooth everywhere (better optimization)", "Zero-centered for negative inputs", "No dead neuron problem"],
        "cons": ["Slower than ReLU (exponential)", "α hyperparameter to tune", "Saturates for large negative values"],
        "fun_fact": "The authors of ELU showed it could match Batch Normalization's benefits in some cases — meaning fewer tricks needed to train deep nets!",
        "code_np": "np.where(x > 0, x, alpha * (np.exp(x) - 1))",
        "code_tf": "tf.keras.activations.elu(x)",
        "code_pt": "torch.nn.ELU()(x)",
    },
    "GELU": {
        "formula": r"f(x) = x \cdot \Phi(x)",
        "desc": "ChatGPT's favorite. Weighs inputs by probability — extremely smooth and expressive.",
        "how_it_works": """
            GELU asks: **"How likely is this input to be positive?"** (using a Gaussian distribution).
            Then it multiplies the input by that probability.
            
            Large positive values? Almost certainly positive → passes through fully.
            Near zero? About 50/50 → gets scaled by half.
            Large negative? Almost certainly not positive → nearly blocked.
            
            This creates an incredibly smooth, probabilistic gate — which is why Transformers love it.
        """,
        "when_to_use": "**Transformers, BERT, GPT, ViT** — the default for modern language and vision models.",
        "when_not_to_use": "Simple/small networks where ReLU is sufficient — GELU's compute overhead isn't worth it.",
        "pros": ["Smoothest activation available", "Probabilistic interpretation", "State-of-the-art for Transformers"],
        "cons": ["Computationally expensive", "Harder to understand intuitively", "Approximation needed for speed"],
        "fun_fact": "GELU was proposed in 2016 but was largely ignored until BERT (2018) and GPT-2 (2019) used it. Now it's in virtually every Transformer model!",
        "code_np": "0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))",
        "code_tf": "tf.keras.activations.gelu(x)",
        "code_pt": "torch.nn.GELU()(x)",
    },
    "Swish": {
        "formula": r"f(x) = x \cdot \sigma(x)",
        "desc": "Google's invention. Looks like ReLU with a smooth dip — often outperforms it in deep nets.",
        "how_it_works": """
            Swish multiplies `x` by its own Sigmoid value: `x × sigmoid(x)`.
            
            The clever part? For large positive x, sigmoid(x) ≈ 1, so output ≈ x (like ReLU).
            For large negative x, sigmoid(x) ≈ 0, so output ≈ 0 (like ReLU).
            But near zero, there's a **smooth dip below zero** — this non-monotonic behavior gives the network extra flexibility.
        """,
        "when_to_use": "**Very deep networks** (EfficientNet, MobileNet). Google found it via automated search and it consistently outperforms ReLU.",
        "when_not_to_use": "When computational budget is tight — it's slower than ReLU due to the Sigmoid computation.",
        "pros": ["Self-gating mechanism", "Smooth and non-monotonic", "Outperforms ReLU in deep nets"],
        "cons": ["Slower than ReLU", "Non-monotonic (can be unexpected)", "Unbounded above"],
        "fun_fact": "Swish was discovered by Google's AutoML! They used a neural network to search for the best activation function — and the AI designed Swish all by itself.",
        "code_np": "x * (1 / (1 + np.exp(-x)))",
        "code_tf": "tf.keras.activations.swish(x)",
        "code_pt": "torch.nn.SiLU()(x)",
    },
    "Softmax": {
        "formula": r"f(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}",
        "desc": "The competition judge — converts raw scores into probabilities that sum to exactly 1.0.",
        "how_it_works": """
            Imagine a **talent show judge** who must distribute exactly 100% of praise among contestants.
            
            Each raw score gets exponentiated (so larger scores get disproportionately bigger),
            then divided by the total. The result: every contestant gets a percentage, and they all add up to 100%.
            
            Higher raw score = much higher probability (the exponential makes differences dramatic).
        """,
        "when_to_use": "**Multi-class classification output layers** — image classification (cat/dog/bird), sentiment analysis (positive/neutral/negative).",
        "when_not_to_use": "**Never in hidden layers!** It forces outputs to sum to 1, killing gradient diversity. Also bad for binary classification (use Sigmoid instead).",
        "pros": ["Perfect probability distribution", "Differentiable", "Works well with cross-entropy loss"],
        "cons": ["Only for output layers", "Numerically unstable without tricks", "All outputs are coupled (changing one changes all)"],
        "fun_fact": "The 'temperature' trick in ChatGPT is just dividing logits by a number before Softmax! Temperature=0.1 makes the model very confident; Temperature=2.0 makes it creative and random.",
        "code_np": "np.exp(x) / np.sum(np.exp(x))",
        "code_tf": "tf.keras.activations.softmax(x)",
        "code_pt": "torch.nn.Softmax(dim=-1)(x)",
    },
}

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">03 / 10 &nbsp;·&nbsp; Interactive</div>
            <h1>Activation Function<br>Playground</h1>
            <p style="max-width:500px;">Explore every activation function interactively — tweak inputs, view outputs, and understand the intuition.</p>
        </div>
        <div class="page-divider"></div>
        <div class="glass-card animate-up delay-1" style="margin-bottom:2rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--stone-900);">💡 How To Use This Page</h3>
            <p style="margin-bottom:0; font-size:0.9rem;">
                Click each tab to explore a different activation function. For each one you'll find:<br>
                <b>📐 The formula</b> — exact math definition &nbsp;│&nbsp;
                <b>🧠 How it works</b> — intuitive explanation &nbsp;│&nbsp;
                <b>🎯 When to use</b> — practical advice &nbsp;│&nbsp;
                <b>📊 Live graph</b> — interactive visualization
            </p>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(list(ACTIVATION_FUNCTIONS.keys()))
    x_range = np.linspace(-6, 6, 200)

    for i, (name, data) in enumerate(ACTIVATION_FUNCTIONS.items()):
        with tabs[i]:
            info = EXPLANATIONS[name]
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                st.markdown(f"<h3 style='margin-bottom:0.3rem;'>{name}</h3>", unsafe_allow_html=True)
                st.latex(info["formula"])

                st.markdown(f"""
                    <div class="glass-card animate-up" style="margin:0.75rem 0;">
                        <div class="section-label" style="margin-bottom:0.2rem;">In Plain English</div>
                        <p style="margin:0; font-size:0.88rem;">{info['desc']}</p>
                    </div>
                """, unsafe_allow_html=True)

                x_val = st.slider(f"Test Input (x)", -6.0, 6.0, 1.0, 0.1, key=f"slider_{name}")

                if name == "Softmax":
                    sample_arr = np.array([-2.0, 0.0, x_val, 2.0])
                    y_val_arr  = data["func"](sample_arr)
                    y_val  = y_val_arr[2]
                    dy_val = data["deriv"](sample_arr)[2]
                else:
                    y_val  = data["func"](x_val)
                    dy_val = data["deriv"](x_val)

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("x", f"{x_val:.2f}")
                mc2.metric("f(x)", f"{y_val:.2f}")
                mc3.metric("f'(x)", f"{dy_val:.2f}")

                st.markdown("<div class='section-label' style='margin-top:1rem;'>Code</div>", unsafe_allow_html=True)
                c1, c2, c3 = st.tabs(["NumPy", "TensorFlow", "PyTorch"])
                c1.code(info["code_np"], language="python")
                c2.code(info["code_tf"], language="python")
                c3.code(info["code_pt"], language="python")

            with col2:
                fig = go.Figure()
                if name == "Softmax":
                    fig.add_trace(go.Bar(x=['x1','x2','x3 (yours)','x4'], y=sample_arr,
                                         name="Logits", marker_color='rgba(168,160,148,0.7)'))
                    fig.add_trace(go.Bar(x=['x1','x2','x3 (yours)','x4'], y=y_val_arr,
                                         name="Probabilities", marker_color='#c8902a'))
                    fig.update_layout(barmode='group',
                                       title=dict(text="Softmax: scores → probabilities (sum=1)",
                                                  font=dict(color='#1a1714', family='Syne', size=13)))
                else:
                    y_plot  = data["func"](x_range)
                    dy_plot = data["deriv"](x_range)
                    fig.add_trace(go.Scatter(x=x_range, y=y_plot, mode='lines', name='f(x)',
                                             line=dict(color='#1a1714', width=2.5)))
                    fig.add_trace(go.Scatter(x=x_range, y=dy_plot, mode='lines', name="f'(x) gradient",
                                             line=dict(color='#c8902a', width=2, dash='dash')))
                    fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers', name='Point',
                                             marker=dict(color='#c8902a', size=12,
                                                         line=dict(color='chocolate', width=2))))
                    fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.1)")
                    fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.1)")

                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.5)",
                    margin=dict(l=20,r=20,t=40,b=20), height=420,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                                font=dict(color='#1a1714', size=11)),
                    xaxis=dict(gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(gridcolor='rgba(0,0,0,0.06)', zerolinecolor='rgba(0,0,0,0.1)'),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── How It Works explanation ──
                with st.expander("🧠 How It Works — Intuitive Explanation", expanded=True):
                    st.markdown(info["how_it_works"])

                # ── When to Use / Not Use ──
                wc1, wc2 = st.columns(2)
                with wc1:
                    st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid #1a1714; min-height:120px;">
                            <div class="section-label" style="margin-bottom:0.2rem;">✅ When To Use</div>
                            <p style="margin:0; font-size:0.83rem;">{info['when_to_use']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with wc2:
                    st.markdown(f"""
                        <div class="glass-card" style="border-left:3px solid #e05d44; min-height:120px;">
                            <div class="section-label" style="margin-bottom:0.2rem;">⚠️ When NOT To Use</div>
                            <p style="margin:0; font-size:0.83rem;">{info['when_not_to_use']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                # ── Pros and Cons ──
                pc1, pc2 = st.columns(2)
                with pc1:
                    pros_html = "".join([f"<li style='font-size:0.82rem;'>{p}</li>" for p in info["pros"]])
                    st.markdown(f"""
                        <div class="glass-card" style="margin-top:0.75rem;">
                            <div class="section-label" style="margin-bottom:0.3rem;">👍 Pros</div>
                            <ul style="margin:0; padding-left:1.2rem;">{pros_html}</ul>
                        </div>
                    """, unsafe_allow_html=True)
                with pc2:
                    cons_html = "".join([f"<li style='font-size:0.82rem;'>{c}</li>" for c in info["cons"]])
                    st.markdown(f"""
                        <div class="glass-card" style="margin-top:0.75rem;">
                            <div class="section-label" style="margin-bottom:0.3rem;">👎 Cons</div>
                            <ul style="margin:0; padding-left:1.2rem;">{cons_html}</ul>
                        </div>
                    """, unsafe_allow_html=True)

                # ── Fun Fact ──
                st.markdown(f"""
                    <div class="glass-card" style="margin-top:0.75rem; border-left:3px solid var(--amber);">
                        <p style="margin:0; font-size:0.85rem;">🎯 <b>Fun Fact:</b> {info['fun_fact']}</p>
                    </div>
                """, unsafe_allow_html=True)

                # ── Warning cards for specific functions ──
                if name == "ReLU":
                    st.markdown("""
                        <div class="glass-card" style="border-left:3px solid #e05d44; margin-top:0.75rem;">
                            <b>⚠️ Dead Neuron Risk:</b> If input is always negative, gradient = 0 forever — neuron stops learning. Try setting the slider to -6 and watch the gradient!
                        </div>""", unsafe_allow_html=True)
                elif name in ["Sigmoid", "Tanh"]:
                    st.markdown("""
                        <div class="glass-card" style="border-left:3px solid #e05d44; margin-top:0.75rem;">
                            <b>⚠️ Saturation Zone:</b> At extreme values the gradient (dashed line) approaches 0 — this is vanishing gradients! Move the slider to ±5 to see it happen.
                        </div>""", unsafe_allow_html=True)
