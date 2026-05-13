import streamlit as st
import random
import plotly.graph_objects as go
import numpy as np

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">09 / 10 &nbsp;·&nbsp; Challenge</div>
            <h1>Challenge<br>Zone</h1>
            <p style="max-width:500px;">Test your knowledge across four games — earn XP and unlock badges.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    # XP indicator
    st.markdown(f"""
        <div class="glass-card animate-up delay-1" style="margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:1rem;">
                <span class="xp-badge" style="font-size:0.85rem;">⭐ {st.session_state.xp} XP</span>
                {' '.join(st.session_state.badges) if st.session_state.badges else
                 '<span style="font-size:0.8rem; color:var(--stone-400);">No badges yet — complete games to earn them!</span>'}
            </div>
            <div style="margin-top:0.75rem;">
                <p style="margin:0; font-size:0.82rem; color:var(--stone-400);">
                    🏅 Earn <b>10 XP</b> per correct curve guess &nbsp;│&nbsp;
                    🏅 <b>20 XP</b> for matching use cases &nbsp;│&nbsp;
                    🏅 <b>15 XP</b> per quiz answer &nbsp;│&nbsp;
                    🏅 <b>25 XP</b> for scenario challenges
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["🎯 Curve Guesser", "🔗 Match Use Case", "🏛️ Architecture Quiz", "🧠 Scenario Challenge"])

    # ── Game 1: Curve Guesser ─────────────────────────────────────
    with tabs[0]:
        st.markdown("""
            <div class="section-label">Game 1 — Visual Recognition</div>
            <h2 style="margin-bottom:0.5rem;">Mystery Curve</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
                <p style="margin:0; font-size:0.88rem;">
                    🔍 <b>How to play:</b> Look at the curve shape below. Each activation function has a unique visual signature:<br>
                    • <b>ReLU</b> = sharp corner at zero, flat on the left<br>
                    • <b>Sigmoid</b> = smooth S-curve between 0 and 1<br>
                    • <b>Tanh</b> = smooth S-curve between -1 and 1
                </p>
            </div>
        """, unsafe_allow_html=True)

        funcs = {
            "ReLU":    lambda x: np.maximum(0, x),
            "Sigmoid": lambda x: 1 / (1 + np.exp(-x)),
            "Tanh":    lambda x: np.tanh(x),
        }

        if 'mystery_func' not in st.session_state:
            st.session_state.mystery_func = random.choice(list(funcs.keys()))

        x = np.linspace(-5, 5, 100)
        y = funcs[st.session_state.mystery_func](x)

        fig = go.Figure(go.Scatter(x=x, y=y, mode='lines',
                                    line=dict(color='#1a1714', width=3)))
        fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.1)")
        fig.add_hline(y=0, line_width=1, line_color="rgba(0,0,0,0.1)")
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.6)",
            xaxis=dict(gridcolor='rgba(0,0,0,0.06)'), yaxis=dict(gridcolor='rgba(0,0,0,0.06)'),
            height=300, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        guess = st.radio("Which activation function is this?",
                         ["Select…"] + list(funcs.keys()), key="g1_guess")
        if st.button("Check Answer", key="btn_g1"):
            if guess == st.session_state.mystery_func:
                st.markdown("""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    🎉 <b>Correct! +10 XP</b> — You've got a great eye for activation shapes!</div>""", unsafe_allow_html=True)
                st.session_state.xp += 10
                st.session_state.mystery_func = random.choice(list(funcs.keys()))
                st.rerun()
            elif guess != "Select…":
                # Give a hint
                hints = {
                    "ReLU": "Look for the sharp corner at x=0 and a perfectly flat left half.",
                    "Sigmoid": "Look for a smooth S-shape that stays between 0 and 1.",
                    "Tanh": "Look for a smooth S-shape that goes from -1 to +1 (crosses zero).",
                }
                st.markdown(f"""<div class="glass-card" style="border-left:3px solid #e05d44;">
                    ❌ Not quite! <b>Hint:</b> {hints[st.session_state.mystery_func]}</div>""",
                    unsafe_allow_html=True)

    # ── Game 2: Match Use Case ────────────────────────────────────
    with tabs[1]:
        st.markdown("""
            <div class="section-label">Game 2 — Practical Knowledge</div>
            <h2 style="margin-bottom:0.5rem;">Match the Use Case</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
                <p style="margin:0; font-size:0.88rem;">
                    🎯 <b>How to play:</b> Match each scenario to the correct activation function.
                    Think about what each function does best — output range, gradient behavior, and speed.
                </p>
            </div>
        """, unsafe_allow_html=True)

        q1 = st.selectbox("Output layer for binary classification (e.g., spam detection)?",
                           ["Select…","ReLU","Sigmoid","Softmax","Tanh"], key="g2_q1")
        q2 = st.selectbox("Hidden layers in deep CNNs (e.g., image recognition)?",
                           ["Select…","ReLU","Sigmoid","Softmax","Tanh"], key="g2_q2")
        q3 = st.selectbox("Multi-class output layer (e.g., cat vs dog vs bird)?",
                           ["Select…","ReLU","Sigmoid","Softmax","Tanh"], key="g2_q3")
        q4 = st.selectbox("LSTM/RNN hidden states (e.g., text generation)?",
                           ["Select…","ReLU","Sigmoid","Softmax","Tanh"], key="g2_q4")

        if st.button("Check All Answers", key="btn_g2"):
            correct = 0
            feedback = []

            if q1 == "Sigmoid":
                correct += 1
                feedback.append("✅ Binary classification → **Sigmoid** (outputs 0-1 probability)")
            else:
                feedback.append("❌ Binary classification → **Sigmoid** squashes output to 0-1 for yes/no decisions")

            if q2 == "ReLU":
                correct += 1
                feedback.append("✅ CNN hidden layers → **ReLU** (fast, no vanishing gradients)")
            else:
                feedback.append("❌ CNN hidden layers → **ReLU** is the default for its speed and stable gradients")

            if q3 == "Softmax":
                correct += 1
                feedback.append("✅ Multi-class output → **Softmax** (outputs sum to 1.0)")
            else:
                feedback.append("❌ Multi-class output → **Softmax** converts scores into probabilities that sum to 1")

            if q4 == "Tanh":
                correct += 1
                feedback.append("✅ LSTM states → **Tanh** (zero-centered, bounded to ±1)")
            else:
                feedback.append("❌ LSTM states → **Tanh** keeps memory values bounded between -1 and 1")

            if correct == 4:
                st.markdown("""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    🏆 <b>Perfect score! +20 XP — 🧠 Badge unlocked!</b></div>""", unsafe_allow_html=True)
                st.session_state.xp += 20
                if "🧠" not in st.session_state.badges:
                    st.session_state.badges.append("🧠")
                    st.balloons()
            elif correct >= 2:
                st.markdown(f"""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    👍 <b>{correct}/4 correct! +{correct * 5} XP</b> — Good effort, review the ones you missed.</div>""", unsafe_allow_html=True)
                st.session_state.xp += correct * 5
            else:
                st.markdown(f"""<div class="glass-card" style="border-left:3px solid #e05d44;">
                    😓 <b>{correct}/4 correct.</b> Review the Playground page to learn each function's role!</div>""", unsafe_allow_html=True)

            for fb in feedback:
                st.markdown(fb)

    # ── Game 3: Architecture Quiz ─────────────────────────────────
    with tabs[2]:
        st.markdown("""
            <div class="section-label">Game 3 — Deep Understanding</div>
            <h2 style="margin-bottom:0.5rem;">Architecture Quiz</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
                <p style="margin:0; font-size:0.88rem;">
                    🧠 <b>How to play:</b> These questions test deeper understanding — not just memorization.
                    Think about <i>why</i> each activation has the properties it does.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Question 1
        st.markdown("<h3 style='margin-bottom:0.5rem;'>Question 1</h3>", unsafe_allow_html=True)
        q3a = st.radio("Why is GELU preferred over ReLU in Transformers?",
                     ["Select…",
                      "It is faster to compute.",
                      "It weights inputs by their probability, creating a smoother curve.",
                      "It completely prevents dead neurons.",
                      "It has a wider output range."],
                     key="g3_q1")

        if st.button("Submit Q1", key="btn_g3_q1"):
            if "weights inputs" in q3a:
                st.markdown("""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    ✅ <b>Correct! +15 XP</b> — GELU's Φ(x) is a Gaussian CDF, giving it a natural probabilistic interpretation
                    that creates smoother, richer representations than ReLU's hard cutoff.</div>""", unsafe_allow_html=True)
                st.session_state.xp += 15
            elif q3a != "Select…":
                st.markdown("""<div class="glass-card" style="border-left:3px solid #e05d44;">
                    ❌ <b>Not quite.</b> GELU isn't faster (it's actually slower), and it doesn't completely prevent dead neurons.
                    The key insight is the probabilistic gating — Φ(x) represents the probability of x being positive under a Gaussian distribution.</div>""",
                    unsafe_allow_html=True)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

        # Question 2
        st.markdown("<h3 style='margin-bottom:0.5rem;'>Question 2</h3>", unsafe_allow_html=True)
        q3b = st.radio("What happens if you use Softmax in hidden layers instead of the output layer?",
                     ["Select…",
                      "The network trains faster.",
                      "Nothing changes — Softmax works fine anywhere.",
                      "Gradient diversity is killed — all neurons compete, slowing training.",
                      "The output probabilities become more accurate."],
                     key="g3_q2")

        if st.button("Submit Q2", key="btn_g3_q2"):
            if "Gradient diversity" in q3b:
                st.markdown("""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    ✅ <b>Correct! +15 XP</b> — Softmax forces all outputs to sum to 1, meaning if one neuron's
                    output increases, all others must decrease. This coupling destroys the independence that
                    hidden layers need to learn diverse features.</div>""", unsafe_allow_html=True)
                st.session_state.xp += 15
            elif q3b != "Select…":
                st.markdown("""<div class="glass-card" style="border-left:3px solid #e05d44;">
                    ❌ <b>Think about it:</b> Softmax makes all outputs sum to 1. In a hidden layer, this means
                    neurons can't learn independently — they're all coupled. This kills gradient diversity
                    and drastically slows learning.</div>""", unsafe_allow_html=True)

        st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)

        # Question 3
        st.markdown("<h3 style='margin-bottom:0.5rem;'>Question 3</h3>", unsafe_allow_html=True)
        q3c = st.radio("Why is Sigmoid's output 'not zero-centered' a problem?",
                     ["Select…",
                      "It makes the output harder to interpret.",
                      "It causes all gradient updates to go in the same direction, creating a zig-zag path.",
                      "It makes the network unable to learn negative values.",
                      "It increases computational cost."],
                     key="g3_q3")

        if st.button("Submit Q3", key="btn_g3_q3"):
            if "zig-zag" in q3c:
                st.markdown("""<div class="glass-card" style="border-left:3px solid var(--amber);">
                    ✅ <b>Correct! +15 XP</b> — When all inputs to a neuron are positive (as Sigmoid outputs are),
                    the gradient for all weights will have the same sign. This means all weights must either
                    increase together or decrease together — the optimizer can't go diagonally, creating an
                    inefficient zig-zag path to the optimum. Tanh fixes this by being zero-centered.</div>""", unsafe_allow_html=True)
                st.session_state.xp += 15
            elif q3c != "Select…":
                st.markdown("""<div class="glass-card" style="border-left:3px solid #e05d44;">
                    ❌ <b>The key issue is gradient direction.</b> When Sigmoid outputs only positive values, the
                    gradients for all weights in the next layer will have the same sign — forcing all weights
                    to update in the same direction. This creates an inefficient zig-zag optimization path.</div>""",
                    unsafe_allow_html=True)

    # ── Game 4: Scenario Challenge ────────────────────────────────
    with tabs[3]:
        st.markdown("""
            <div class="section-label">Game 4 — Real-World Decisions</div>
            <h2 style="margin-bottom:0.5rem;">Scenario Challenge</h2>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card" style="margin-bottom:1rem; padding:0.9rem 1.2rem;">
                <p style="margin:0; font-size:0.88rem;">
                    🎯 <b>How to play:</b> You're the ML engineer. Read each scenario and decide which
                    activation function to use. Think about the data type, output requirements, and network depth.
                </p>
            </div>
        """, unsafe_allow_html=True)

        scenarios = [
            {
                "scenario": "You're building a 152-layer ResNet for image classification. You need the hidden layers to train fast without vanishing gradients. Which activation?",
                "options": ["Sigmoid", "ReLU", "Softmax", "Tanh"],
                "answer": "ReLU",
                "explanation": "ReLU's gradient is 1 for positive inputs — perfect for very deep networks. Sigmoid/Tanh would cause vanishing gradients at 152 layers. This is exactly what ResNet uses!"
            },
            {
                "scenario": "You're building a GPT-style language model with 96 Transformer layers. The research papers recommend a specific activation for the feed-forward blocks. Which one?",
                "options": ["ReLU", "Sigmoid", "GELU", "Leaky ReLU"],
                "answer": "GELU",
                "explanation": "GELU is the standard for Transformers since BERT/GPT-2. Its smooth, probabilistic gating creates richer representations than ReLU for self-attention-based architectures."
            },
            {
                "scenario": "You're building a self-driving car's object detection system. Some neurons in your network are 'dying' (permanently outputting zero) and missing pedestrians. How do you fix this?",
                "options": ["Add more data", "Use Leaky ReLU instead of ReLU", "Use Softmax", "Increase learning rate"],
                "answer": "Use Leaky ReLU instead of ReLU",
                "explanation": "Dead neurons happen when ReLU permanently outputs 0. Leaky ReLU's small negative slope (0.01x for x < 0) ensures every neuron can recover. This is why YOLO uses Leaky ReLU!"
            },
        ]

        for j, s in enumerate(scenarios):
            st.markdown(f"<h3 style='margin-bottom:0.3rem;'>Scenario {j+1}</h3>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class="glass-card" style="margin-bottom:0.75rem;">
                    <p style="margin:0; font-size:0.9rem;">{s['scenario']}</p>
                </div>
            """, unsafe_allow_html=True)

            answer = st.radio("Your choice:", ["Select…"] + s["options"], key=f"scenario_{j}")

            if st.button(f"Submit Scenario {j+1}", key=f"btn_scenario_{j}"):
                if answer == s["answer"]:
                    st.markdown(f"""<div class="glass-card" style="border-left:3px solid var(--amber);">
                        ✅ <b>Correct! +25 XP</b> — {s['explanation']}</div>""", unsafe_allow_html=True)
                    st.session_state.xp += 25
                    if st.session_state.xp >= 50 and "🏆" not in st.session_state.badges:
                        st.session_state.badges.append("🏆")
                        st.balloons()
                elif answer != "Select…":
                    st.markdown(f"""<div class="glass-card" style="border-left:3px solid #e05d44;">
                        ❌ <b>Not quite.</b> {s['explanation']}</div>""", unsafe_allow_html=True)

            if j < len(scenarios) - 1:
                st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
