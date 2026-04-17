import json
import html
import re
import pandas as pd
import streamlit as st

from src.aspect_mapper import ASPECT_CONFIG
from src.frontend_service import load_model, analyze_single_comment, run_batch_analysis


st.set_page_config(
    page_title="Hybrid Explainable Sentiment Analyzer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 3.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.sub-title {
    color: #5b6470;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.info-box {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 1rem;
}

.section-title {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.6rem;
    margin-bottom: 0.5rem;
}

.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}

.download-box {
    background: #eef6ff;
    border: 1px solid #cfe4ff;
    padding: 1rem;
    border-radius: 14px;
}

.explanation-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 1rem 1.1rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    line-height: 1.8;
    font-size: 1rem;
}

.toprule-box {
    background: #f6f8fb;
    border: 1px solid #e6ebf2;
    padding: 0.95rem 1.05rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 1.02rem;
    line-height: 1.8;
    color: #1f2937;
    white-space: pre-wrap;
    word-break: break-word;
}

.highlight-label-positive {
    background: #e8f7ec;
    color: #18794e;
    padding: 0.08rem 0.38rem;
    border-radius: 8px;
    font-weight: 600;
}

.highlight-label-negative {
    background: #fdecec;
    color: #c0392b;
    padding: 0.08rem 0.38rem;
    border-radius: 8px;
    font-weight: 600;
}

.highlight-label-neutral {
    background: #eef2f7;
    color: #4b5563;
    padding: 0.08rem 0.38rem;
    border-radius: 8px;
    font-weight: 600;
}

.highlight-aspect {
    background: #eef4ff;
    color: #1d4ed8;
    padding: 0.08rem 0.38rem;
    border-radius: 8px;
    font-weight: 600;
}

.highlight-rule-low {
    color: #2563eb;
    font-weight: 700;
}

.highlight-rule-medium {
    color: #d97706;
    font-weight: 700;
}

.highlight-rule-high {
    color: #dc2626;
    font-weight: 700;
}

.highlight-rule-positive {
    color: #15803d;
    font-weight: 700;
}

.highlight-rule-negative {
    color: #dc2626;
    font-weight: 700;
}

.highlight-rule-neutral {
    color: #6b7280;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 0.8rem 1rem;
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}

hr {
    margin-top: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


def highlight_explanation_text(text: str) -> str:
    text = html.escape(str(text))

    replacements = [
        (r"\bpositive\b", '<span class="highlight-label-positive">positive</span>'),
        (r"\bnegative\b", '<span class="highlight-label-negative">negative</span>'),
        (r"\bneutral\b", '<span class="highlight-label-neutral">neutral</span>'),

        (r"\bgeneral\b", '<span class="highlight-aspect">general</span>'),
        (r"\bperformance\b", '<span class="highlight-aspect">performance</span>'),
        (r"\bservice\b", '<span class="highlight-aspect">service</span>'),
        (r"\bpricing\b", '<span class="highlight-aspect">pricing</span>'),
        (r"\bfeature\b", '<span class="highlight-aspect">feature</span>'),
        (r"\busability\b", '<span class="highlight-aspect">usability</span>'),
        (r"\binterface\b", '<span class="highlight-aspect">interface</span>'),
        (r"\bquality\b", '<span class="highlight-aspect">quality</span>'),
        (r"\bfit_size\b", '<span class="highlight-aspect">fit_size</span>'),
        (r"\bstyle_appearance\b", '<span class="highlight-aspect">style_appearance</span>'),
        (r"\bcomfort\b", '<span class="highlight-aspect">comfort</span>'),
        (r"\bdelivery\b", '<span class="highlight-aspect">delivery</span>'),
        (r"\breliability\b", '<span class="highlight-aspect">reliability</span>'),
        (r"\baccount_billing\b", '<span class="highlight-aspect">account_billing</span>'),
        (r"\bsecurity_privacy\b", '<span class="highlight-aspect">security_privacy</span>'),
        (r"\bcontent\b", '<span class="highlight-aspect">content</span>'),
        (r"\boverall_experience\b", '<span class="highlight-aspect">overall_experience</span>'),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def highlight_top_rule(rule: str) -> str:
    rule = html.escape(str(rule))

    replacements = [
        (r"\blow\b", '<span class="highlight-rule-low">low</span>'),
        (r"\bmedium\b", '<span class="highlight-rule-medium">medium</span>'),
        (r"\bhigh\b", '<span class="highlight-rule-high">high</span>'),

        (r"\bpositive\b", '<span class="highlight-rule-positive">positive</span>'),
        (r"\bnegative\b", '<span class="highlight-rule-negative">negative</span>'),
        (r"\bneutral\b", '<span class="highlight-rule-neutral">neutral</span>'),
    ]

    for pattern, replacement in replacements:
        rule = re.sub(pattern, replacement, rule, flags=re.IGNORECASE)

    return rule


@st.cache_resource
def get_model_bundle():
    return load_model()


model, tokenizer, device = get_model_bundle()

with st.sidebar:
    st.header("Workflow")
    st.markdown("""
1. Choose a tab  
2. Enter one comment or upload a CSV  
3. Run analysis  
4. Review labels, aspects, evidence, and export results  
""")
    st.header("Current System")
    st.markdown("""
- BERT sentiment probabilities  
- Tsukamoto fuzzy inference  
- Evidence phrase extraction  
- Configurable aspect mapping  
""")
    st.caption("Batch outputs are saved under `outputs/frontend_outputs/`.")


st.markdown('<div class="main-title">Hybrid Explainable Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">BERT + Tsukamoto Fuzzy + Evidence Phrase + Aspect Mapping</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
<div class="info-box">
This system supports <b>single-comment sentiment analysis</b> and <b>batch CSV review analysis</b>.
It provides not only sentiment labels, but also <b>rule-based explanations</b>, <b>evidence phrases</b>,
and <b>aspect-level interpretation</b>.
</div>
""",
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "Single Comment Analysis",
    "Batch CSV Analysis",
    "Aspect Configuration"
])

with tab1:
    st.markdown('<div class="section-title">Single Comment Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Enter one comment to analyze its sentiment, aspect, fuzzy rule, explanation, and evidence phrases.</div>',
        unsafe_allow_html=True
    )

    text_input = st.text_area(
        "Comment",
        height=160,
        placeholder="Example: The design is beautiful but the zipper is hard to use."
    )

    if st.button("Analyze Comment", type="primary"):
        if not text_input.strip():
            st.warning("Please enter a comment first.")
        else:
            result = analyze_single_comment(text_input, model, tokenizer, device)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Label", result["final_label"])
            c2.metric("Aspect", result["aspect"])
            c3.metric("Risk Level", result["risk_level"])
            c4.metric("Score", f"{result['final_score']:.4f}")

            left, right = st.columns([1.3, 1])

            with left:
                st.markdown("### Explanation")
                st.markdown(
                    f'<div class="explanation-box">{highlight_explanation_text(result["explanation"])}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("### Action Hint")
                st.info(result["action_hint"])

                st.markdown("### Top Rule")
                st.markdown(
                    f'<div class="toprule-box">{highlight_top_rule(result["top_rule"])}</div>',
                    unsafe_allow_html=True
                )

            with right:
                st.markdown("### BERT Probabilities")
                prob_df = pd.DataFrame([{
                    "p_neg": round(result["p_neg"], 4),
                    "p_neu": round(result["p_neu"], 4),
                    "p_pos": round(result["p_pos"], 4)
                }])
                st.dataframe(prob_df, use_container_width=True, hide_index=True)

            st.markdown("### Evidence Phrases")
            if result["evidence_rows"]:
                evidence_df = pd.DataFrame(result["evidence_rows"])
                st.dataframe(evidence_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No strong evidence phrase extracted for this comment.")

with tab2:
    st.markdown('<div class="section-title">Batch CSV Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Upload a CSV file containing a <code>comment</code> column. Optional columns such as <code>review_id</code>, <code>score</code>, and <code>reference_label</code> are supported.</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("### Preview")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run Batch Analysis", type="primary"):
            with st.spinner("Running batch analysis..."):
                result_df, summary_df, evidence_summary_df, excel_file, figures_dir, dataset_name, output_dir = run_batch_analysis(
                    df, uploaded_file.name, model, tokenizer, device
                )

            st.success(f"Batch analysis completed. Output folder: {output_dir}")

            summary_map = dict(zip(summary_df["metric"], summary_df["value"]))

            st.markdown("### Key Metrics")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Negative", f"{summary_map.get('negative_proportion', 0)*100:.1f}%")
            with k2:
                st.metric("Neutral", f"{summary_map.get('neutral_proportion', 0)*100:.1f}%")
            with k3:
                st.metric("Positive", f"{summary_map.get('positive_proportion', 0)*100:.1f}%")
            with k4:
                st.metric("Match Rate", f"{summary_map.get('overall_match_rate', 0)*100:.1f}%")

            with st.expander("View Summary Table", expanded=False):
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("### Analysis Tables")
            col_left, col_right = st.columns([1, 1])

            with col_left:
                with st.expander("Evidence Summary", expanded=True):
                    st.dataframe(evidence_summary_df, use_container_width=True, hide_index=True)

            with col_right:
                with st.expander("Analyzed Comments", expanded=False):
                    st.dataframe(result_df.head(50), use_container_width=True, hide_index=True)

            st.markdown("### Visual Dashboard")

            pie_path = figures_dir / "label_distribution_pie.png"
            neg_path = figures_dir / "negative_keywords_bar.png"
            neu_path = figures_dir / "neutral_keywords_bar.png"
            pos_path = figures_dir / "positive_keywords_bar.png"

            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                if pie_path.exists():
                    st.image(str(pie_path), caption="Label Distribution", use_container_width=True)
            with row1_col2:
                if neg_path.exists():
                    st.image(str(neg_path), caption="Negative Evidence", use_container_width=True)

            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                if neu_path.exists():
                    st.image(str(neu_path), caption="Neutral Evidence", use_container_width=True)
            with row2_col2:
                if pos_path.exists():
                    st.image(str(pos_path), caption="Positive Evidence", use_container_width=True)

            st.markdown("### Export")
            st.markdown(
                f'<div class="download-box"><b>Saved folder:</b> <code>{output_dir}</code></div>',
                unsafe_allow_html=True
            )

            with open(excel_file, "rb") as f:
                st.download_button(
                    label="Download Excel Report",
                    data=f,
                    file_name=f"{dataset_name}_comment_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

with tab3:
    st.markdown('<div class="section-title">Aspect Configuration</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">The aspect dictionary is stored in <code>src/aspect_mapper.py</code>. You can expand it to support new domains such as e-commerce, SaaS, or product reviews.</div>',
        unsafe_allow_html=True
    )

    pretty_config = {k: sorted(list(v)) for k, v in ASPECT_CONFIG.items()}

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Aspect Dictionary")
        st.json(pretty_config)

    with c2:
        st.markdown("### Copyable JSON")
        st.code(json.dumps(pretty_config, indent=2), language="json")