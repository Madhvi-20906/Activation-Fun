import streamlit as st

def render():
    st.markdown("""
        <div class="animate-up">
            <div class="section-label">07 / 10 &nbsp;·&nbsp; Industry</div>
            <h1>Real World<br>Applications</h1>
            <p style="max-width:500px;">How tech giants use these functions in production — from ChatGPT to self-driving cars.</p>
        </div>
        <div class="page-divider"></div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="glass-card animate-up delay-1" style="margin-bottom:2rem; border-left: 4px solid var(--amber);">
            <h3 style="margin-top:0; color:var(--amber);">🌍 Why This Matters</h3>
            <p style="font-size:0.9rem; margin-bottom:0;">
                Choosing the right activation function isn't just academic — it directly affects whether a product 
                works or fails. Every AI product you use daily was carefully designed with specific activations 
                for specific reasons. Let's see what the industry uses and <b>why</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    apps = [
        {
            "icon": "🖼️", "num": "01",
            "title": "Face Recognition",
            "act": "ReLU",
            "company": "Apple Face ID, Meta, Google Photos",
            "desc": "Used in deep CNN layers like ResNet. Avoids vanishing gradients and enables extremely deep networks to train fast.",
            "why": "Face recognition requires very deep networks (100+ layers) to learn fine-grained features like nose shape, eye spacing, and skin texture. ReLU's constant gradient of 1 for positive values means even the earliest layers (which detect basic edges) receive strong learning signals. Without ReLU, networks this deep would be impossible to train.",
            "architecture": "Input Image → Conv+ReLU × 50+ layers → Embedding → Distance Metric → Match/No Match",
        },
        {
            "icon": "🎬", "num": "02",
            "title": "Netflix Recommendations",
            "act": "GELU",
            "company": "Netflix, Spotify, YouTube",
            "desc": "Powers dense embedding models. Its smoothness helps capture nuanced user-item interaction patterns.",
            "why": "Recommendation systems need to understand subtle preferences: you might like action movies but only on weekends, or prefer jazz when working. GELU's probabilistic gating creates smoother embeddings that capture these nuanced patterns better than ReLU's hard cutoff at zero.",
            "architecture": "User History → Embedding → Transformer+GELU → Attention → Top-K Recommendations",
        },
        {
            "icon": "🗣️", "num": "03",
            "title": "Voice Assistants",
            "act": "Tanh",
            "company": "Siri, Alexa, Google Assistant",
            "desc": "Used in LSTM cell states to keep audio-signal memory tightly bound between −1 and 1.",
            "why": "Audio signals naturally oscillate between positive and negative values. Tanh's range of (-1, 1) is perfect for this — it preserves the sign of the signal while preventing any single memory from dominating. If you used ReLU (which kills negatives), the LSTM would lose half the audio information!",
            "architecture": "Audio Waveform → MFCC Features → LSTM+Tanh (memory) → Attention → Text Output",
        },
        {
            "icon": "🩺", "num": "04",
            "title": "Medical AI (Tumor Detection)",
            "act": "Sigmoid",
            "company": "Google Health, PathAI, Paige",
            "desc": "Final layer outputs a strict 0–1 probability for 'Malignant' vs 'Benign' classification.",
            "why": "In medical AI, you need a clear probability — 'there's a 94% chance this is malignant.' Sigmoid naturally outputs values between 0 and 1, making it the only choice for binary classification output. Doctors can then set their own threshold (e.g., 'flag anything above 80%').",
            "architecture": "Medical Scan → CNN+ReLU (features) → Dense → Sigmoid → Probability (0-1)",
        },
        {
            "icon": "🤖", "num": "05",
            "title": "ChatGPT / Large Language Models",
            "act": "GELU + Softmax",
            "company": "OpenAI, Google, Anthropic, Meta",
            "desc": "GELU powers feed-forward blocks; Softmax converts final scores into next-token probabilities.",
            "why": "GPT uses GELU in its feed-forward layers because the smooth, probabilistic gating creates richer representations than ReLU. Then Softmax at the output converts raw scores into a probability distribution over 50,000+ words — 'next word is 40% likely to be THE, 15% likely to be A, ...' — and the model samples from this distribution.",
            "architecture": "Tokens → Embedding → (Self-Attention → FFN+GELU) × 96 layers → Softmax → Next Token",
        },
        {
            "icon": "🚗", "num": "06",
            "title": "Self-Driving Cars",
            "act": "Leaky ReLU",
            "company": "Tesla, Waymo, Cruise",
            "desc": "Used in YOLO object detection — prevents dead neurons so the car never ignores critical visual features.",
            "why": "In self-driving, missing a pedestrian or stop sign is life-threatening. Standard ReLU can cause 'dead neurons' — neurons that permanently output zero and ignore certain features. Leaky ReLU's small negative slope ensures EVERY neuron stays active and responsive. You never want a neuron responsible for detecting pedestrians to 'die'!",
            "architecture": "Camera Feed → Backbone+LeakyReLU → Feature Pyramid → Detection Head → Bounding Boxes",
        },
    ]

    for i, app in enumerate(apps):
        st.markdown(f"""
            <div class="service-card animate-up" style="margin-bottom:1.5rem;">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <span style="font-size:1.8rem;">{app['icon']}</span>
                        <div>
                            <div class="section-label" style="margin-bottom:0;">{app['num']}</div>
                            <h3 style="margin:0; font-size:1rem;">{app['title']}</h3>
                        </div>
                    </div>
                    <span style="background:var(--stone-900); color:chocolate; font-size:0.68rem;
                                 font-weight:700; letter-spacing:0.08em; padding:3px 10px;
                                 border-radius:99px; white-space:nowrap;">
                        {app['act']}
                    </span>
                </div>
                <p style="margin:0 0 0.3rem 0; font-size:0.78rem; color:var(--stone-400); font-weight:600; letter-spacing:0.05em; text-transform:uppercase;">
                    {app['company']}
                </p>
                <p style="margin:0 0 0.75rem 0; font-size:0.88rem;">{app['desc']}</p>
            </div>
        """, unsafe_allow_html=True)

        with st.expander(f"🔍 Deep Dive: Why {app['act']} for {app['title']}?"):
            st.markdown(f"**Why this activation?**\n\n{app['why']}")
            st.markdown(f"**Typical Architecture:**")
            st.code(app['architecture'], language=None)

    # ── Industry Trends ──────────────────────────────────────────
    st.markdown('<div class="page-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <h2 style="margin-bottom:1rem;">Industry Trends</h2>
    """, unsafe_allow_html=True)

    tc1, tc2, tc3 = st.columns(3, gap="medium")
    trends = [
        ("📈", "2012–2015", "The ReLU Revolution",
         "AlexNet proved ReLU trains 6× faster than Tanh. Within 2 years, virtually every CNN switched to ReLU. Deep networks became practical for the first time."),
        ("🔄", "2016–2019", "The Smoothness Era",
         "GELU and Swish emerged. BERT and GPT proved that smoother activations give better results for Transformers. Google used AutoML to discover Swish automatically."),
        ("🚀", "2020–Present", "Task-Specific Choices",
         "Modern networks mix activations: GELU in Transformers, ReLU in CNNs, Sigmoid/Softmax in output layers. The field has matured — there's no single 'best' function."),
    ]
    for col, (icon, era, title, desc) in zip([tc1, tc2, tc3], trends):
        with col:
            st.markdown(f"""
                <div class="service-card animate-up" style="min-height:220px;">
                    <div style="font-size:1.5rem; margin-bottom:0.5rem;">{icon}</div>
                    <div class="section-label" style="margin-bottom:0.2rem;">{era}</div>
                    <h3 style="margin:0 0 0.4rem 0; font-size:0.85rem;">{title}</h3>
                    <p style="font-size:0.8rem; margin:0;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)
