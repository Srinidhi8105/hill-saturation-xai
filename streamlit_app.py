
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({"font.family": "monospace", "font.size": 10})

st.set_page_config(
    page_title="Hill Saturation XAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e1117;
    color: #e8e8e8;
}
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #242938);
    border: 1px solid #2e3650;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #f97316;
    font-family: "Space Mono", monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #8b92a5;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}
.highlight-box {
    background: linear-gradient(135deg, #1a2340, #1e2a42);
    border-left: 4px solid #f97316;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.9rem;
    color: #cdd3e0;
}
.section-divider {
    border: none;
    border-top: 1px solid #2e3650;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
PROJECT_DIR = "."
DATASET_CSV = "rd_innovation_dataset_with_features.csv"

FEATURES = [
    "rd_spend", "company_size", "industry_maturity",
    "talent_density", "collaboration_index", "prior_patents",
    "industry_encoded", "hill_base_score"
]

INDUSTRY_MAP = {"energy": 0, "manufacturing": 1, "pharma": 2, "tech": 3}
INDUSTRY_COLORS = {
    "tech": "#3b82f6", "pharma": "#22c55e",
    "manufacturing": "#f97316", "energy": "#a855f7"
}

def hill_func(x, K, n):
    return (x ** n) / (K ** n + x ** n)

def style_axis(ax):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#8b92a5")
    for spine in ax.spines.values():
        spine.set_color("#2e3650")

def style_fig(fig):
    fig.patch.set_facecolor("#0e1117")

# ── Load assets ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model and data...")
def load_assets():
    model       = joblib.load(f"{PROJECT_DIR}/hill_gb_model.pkl")
    hill_params = joblib.load(f"{PROJECT_DIR}/hill_params.pkl")
    df          = pd.read_csv(DATASET_CSV)
    return model, hill_params, df

try:
    model, hill_params, df = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Could not load saved assets: {e}")
    st.info(f"Looking for dataset at: {DATASET_CSV}")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 Company Parameters")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    industry            = st.selectbox("Industry", ["tech", "pharma", "manufacturing", "energy"])
    rd_spend            = st.slider("R&D Spend ($M)", 1.0, 500.0, 120.0, step=1.0)
    company_size        = st.slider("Company Size (employees)", 50, 100_000, 5_000, step=50)
    industry_maturity   = st.slider("Industry Maturity", 0.0, 1.0, 0.4, step=0.01)
    talent_density      = st.slider("Talent Density", 0.05, 0.85, 0.45, step=0.01)
    collaboration_index = st.slider("Collaboration Index", 0.0, 1.0, 0.35, step=0.01)
    prior_patents       = st.slider("Prior Patents", 0, 300, 50, step=1)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    page = st.radio("📊 Navigate", [
        "🏠  Overview",
        "🔮  Predict & Explain",
        "🌍  SHAP Global Analysis",
        "📉  Saturation Explorer",
    ])

# ── Header ────────────────────────────────────────────────────
st.markdown("# 📈 Hill Saturation — XAI Dashboard")
st.markdown(
    "<div class='highlight-box'>Explore how R&D investment, talent, and collaboration "
    "drive innovation — and <strong>where spending stops paying off</strong>.</div>",
    unsafe_allow_html=True
)

if not assets_loaded:
    st.stop()

# ── Feature vector builder ────────────────────────────────────
def build_input():
    K   = hill_params[industry]["K"]
    n   = hill_params[industry]["n"]
    hbs = hill_func(rd_spend, K, n) * 100
    return pd.DataFrame([{
        "rd_spend":            rd_spend,
        "company_size":        company_size,
        "industry_maturity":   industry_maturity,
        "talent_density":      talent_density,
        "collaboration_index": collaboration_index,
        "prior_patents":       prior_patents,
        "industry_encoded":    INDUSTRY_MAP[industry],
        "hill_base_score":     hbs,
    }])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (len(df),                               "Companies"),
        (df["industry_type"].nunique(),          "Industries"),
        (f"{df['innovation_score'].mean():.1f}", "Avg Innovation Score"),
        (f"{df['rd_spend'].median():.0f}M",      "Median R&D Spend"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    st.markdown("### R&D Spend vs Innovation Score")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    style_fig(fig)

    for ind, grp in df.groupby("industry_type"):
        axes[0].scatter(
            grp["rd_spend"], grp["innovation_score"],
            alpha=0.35, s=18, color=INDUSTRY_COLORS[ind], label=ind
        )
    style_axis(axes[0])
    axes[0].set_xlabel("R&D Spend ($M)", color="#8b92a5")
    axes[0].set_ylabel("Innovation Score", color="#8b92a5")
    axes[0].set_title("Spend vs Score by Industry", color="#e8e8e8", fontweight="bold")
    axes[0].legend(facecolor="#1a1f2e", labelcolor="#e8e8e8")

    for ind, grp in df.groupby("industry_type"):
        axes[1].hist(grp["innovation_score"], bins=20, alpha=0.6,
                     color=INDUSTRY_COLORS[ind], label=ind)
    style_axis(axes[1])
    axes[1].set_xlabel("Innovation Score", color="#8b92a5")
    axes[1].set_ylabel("Count", color="#8b92a5")
    axes[1].set_title("Score Distribution", color="#e8e8e8", fontweight="bold")
    axes[1].legend(facecolor="#1a1f2e", labelcolor="#e8e8e8")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — Predict & Explain
# ════════════════════════════════════════════════════════════════
elif page == "🔮  Predict & Explain":

    X_input    = build_input()
    prediction = model.predict(X_input)[0]
    K_val      = hill_params[industry]["K"]
    n_val      = hill_params[industry]["n"]
    hbs        = hill_func(rd_spend, K_val, n_val) * 100
    is_over    = rd_spend >= K_val

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{prediction:.1f}</div>"
        f"<div class='metric-label'>Predicted Innovation Score</div></div>",
        unsafe_allow_html=True
    )
    c2.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value'>{hbs:.1f}%</div>"
        f"<div class='metric-label'>Hill Saturation Reached</div></div>",
        unsafe_allow_html=True
    )
    status = "🔴 Over-saturated" if is_over else "🟢 Under-saturated"
    c3.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-value' style='font-size:1.1rem'>{status}</div>"
        f"<div class='metric-label'>vs K = {K_val}M threshold</div></div>",
        unsafe_allow_html=True
    )

    advice = (
        f"You are past the half-saturation point (K = {K_val}M). "
        "Redirecting budget towards <strong>talent</strong> or "
        "<strong>collaboration</strong> is likely more effective than increasing R&D spend."
        if is_over else
        f"You are still in the high-return zone (K = {K_val}M). "
        "Increasing R&D spend should continue to drive meaningful innovation gains."
    )
    st.markdown(f"<div class='highlight-box'>💡 {advice}</div>", unsafe_allow_html=True)

    st.markdown("### SHAP Waterfall — Feature Contributions")
    explainer   = shap.TreeExplainer(model)
    sv          = explainer.shap_values(X_input)
    explanation = shap.Explanation(
        values        = sv[0],
        base_values   = explainer.expected_value,
        data          = X_input.iloc[0].values,
        feature_names = FEATURES
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    shap.waterfall_plot(explanation, show=False, max_display=8)
    ax = plt.gca()
    ax.set_facecolor("#ffffff")
    plt.title("Why did the model predict this score?",
              color="#111111", fontweight="bold", pad=14, fontsize=12)
    plt.tight_layout()
    # Wrap in a styled container
    st.markdown(
        "<div style='background:#ffffff; border-radius:12px; padding:12px;"
        "border:1px solid #2e3650;'>",
        unsafe_allow_html=True
    )
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)
    plt.close()


    st.markdown("### Input Feature Values")
    st.dataframe(X_input.style.format("{:.3f}"), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP Global Analysis
# ════════════════════════════════════════════════════════════════
elif page == "🌍  SHAP Global Analysis":

    st.markdown("### Global SHAP Feature Importance")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    @st.cache_data(show_spinner="Computing SHAP values...")
    def compute_global_shap():
        le  = LabelEncoder()
        df2 = df.copy()
        df2["industry_encoded"] = le.fit_transform(df2["industry_type"])

        def get_hbs(row):
            K = hill_params[row["industry_type"]]["K"]
            n = hill_params[row["industry_type"]]["n"]
            return hill_func(row["rd_spend"], K, n) * 100

        df2["hill_base_score"] = df2.apply(get_hbs, axis=1)
        X_all = df2[FEATURES]
        y_all = df2["innovation_score"]
        _, X_test_g, _, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        explainer_g = shap.TreeExplainer(model)
        sv_g        = explainer_g.shap_values(X_test_g)
        return sv_g, X_test_g

    sv_g, X_test_g = compute_global_shap()

    fig, ax = plt.subplots(figsize=(10, 6))
    style_fig(fig)
    shap.summary_plot(sv_g, X_test_g, feature_names=FEATURES,
                      plot_type="dot", show=False, max_display=8)
    style_axis(plt.gca())
    plt.title("SHAP Beeswarm — Global Feature Impact", color="#e8e8e8",
              fontweight="bold", pad=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Mean |SHAP| — Feature Ranking")
    mean_shap = pd.DataFrame({
        "Feature":     FEATURES,
        "Mean |SHAP|": np.abs(sv_g).mean(axis=0)
    }).sort_values("Mean |SHAP|", ascending=True)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    style_fig(fig2)
    ax2.barh(mean_shap["Feature"], mean_shap["Mean |SHAP|"],
             color="#f97316", alpha=0.85)
    style_axis(ax2)
    ax2.set_xlabel("Mean |SHAP| Value", color="#8b92a5")
    ax2.set_title("Feature Importance Ranking", color="#e8e8e8", fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ════════════════════════════════════════════════════════════════
# PAGE 4 — Saturation Explorer
# ════════════════════════════════════════════════════════════════
elif page == "📉  Saturation Explorer":

    st.markdown("### Hill Saturation Curves by Industry")
    spend_range = np.linspace(1, 500, 300)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    style_fig(fig)

    for ind, cfg in hill_params.items():
        curve = hill_func(spend_range, cfg["K"], cfg["n"]) * 100
        axes[0].plot(spend_range, curve, color=INDUSTRY_COLORS[ind],
                     linewidth=2.5, label=ind)
        axes[0].axvline(cfg["K"], color=INDUSTRY_COLORS[ind],
                        linestyle="--", linewidth=0.8, alpha=0.5)

    K_u        = hill_params[industry]["K"]
    n_u        = hill_params[industry]["n"]
    user_score = hill_func(rd_spend, K_u, n_u) * 100
    axes[0].scatter([rd_spend], [user_score], s=140, color="white",
                    zorder=5, edgecolors="#f97316", linewidths=2.5,
                    label="Your company")
    style_axis(axes[0])
    axes[0].set_xlabel("R&D Spend ($M)", color="#8b92a5")
    axes[0].set_ylabel("Hill Score (0–100)", color="#8b92a5")
    axes[0].set_title("Hill Curves — All Industries", color="#e8e8e8", fontweight="bold")
    axes[0].legend(facecolor="#1a1f2e", labelcolor="#e8e8e8")

    delta    = 5.0
    marginal = np.array([
        (hill_func(s + delta, K_u, n_u) - hill_func(s, K_u, n_u)) / delta * 100
        for s in spend_range
    ])
    axes[1].plot(spend_range, marginal, color="#f97316", linewidth=2.5)
    axes[1].axvline(K_u, color="#e8e8e8", linestyle="--", linewidth=1,
                    label=f"Half-saturation K={K_u}M")
    axes[1].axvline(rd_spend, color="#3b82f6", linestyle="-", linewidth=1.8,
                    label=f"Your spend ${rd_spend:.0f}M")
    axes[1].fill_between(spend_range, marginal,
                         where=(spend_range <= rd_spend),
                         alpha=0.15, color="#3b82f6")
    style_axis(axes[1])
    axes[1].set_xlabel("R&D Spend ($M)", color="#8b92a5")
    axes[1].set_ylabel("Marginal Gain per $M", color="#8b92a5")
    axes[1].set_title(f"Marginal Returns — {industry.title()}", color="#e8e8e8",
                      fontweight="bold")
    axes[1].legend(facecolor="#1a1f2e", labelcolor="#e8e8e8")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Industry Saturation Parameters")
    param_df = pd.DataFrame(hill_params).T.reset_index()
    param_df.columns = ["Industry", "K — Half-Saturation ($M)", "n — Shape"]
    st.dataframe(
        param_df.style.format({
            "K — Half-Saturation ($M)": "{:.1f}",
            "n — Shape": "{:.2f}"
        }),
        use_container_width=True
    )

    advice = (
        f"You are past half-saturation (K = {K_u}M). "
        "Redirecting budget to <strong>talent or collaboration</strong> will yield more innovation."
        if rd_spend >= K_u else
        f"You are in the high-return zone (below K = {K_u}M). "
        "Increasing R&D spend should still drive meaningful gains."
    )
    st.markdown(f"<div class='highlight-box'>💡 {advice}</div>", unsafe_allow_html=True)
