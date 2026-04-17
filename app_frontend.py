import json
from pathlib import Path

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
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.metric-card {
    background-color: #f7f9fc;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #e6eaf0;
}
.section-box {
    background-color: #ffffff;
    padding: 1rem 1rem 0.5rem 1rem;
    border-radius: 14px;
    border: 1px solid #e8edf3;
    margin-bottom: 1rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.92rem;
}
.download-box {
    background-color: #f0f8ff;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #d7ebff;
}
</style>
""", unsafe_allow_html=True)

st.title("Hybrid Explainable Sentiment Analyzer")
st.caption("BERT + Tsukamoto Fuzzy + Evidence Phrase + Aspect Mapping")

st.markdown(
    """
    This interface supports:
    - **single comment analysis**
    - **batch CSV analysis**
    - **configurable aspect inspection**
    """
)

@st.cache_resource
def get_model_bundle():
    return load_model()

model, tokenizer, device = get_model_bundle()

tab1, tab2, tab3 = st.tabs([
    "Single Comment Analysis",
    "Batch CSV Analysis",
    "Aspect Configuration"
])

with tab1:
    st.subheader("Single Comment Analysis")
    st.markdown('<div class="small-note">Enter one review or comment to analyze its sentiment, aspect, rule explanation, and evidence phrases.</div>', unsafe_allow_html=True)

    text_input = st.text_area(
        "Comment",
        height=160,
        placeholder="Type a review or comment here..."
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

            left, right = st.columns([1.2, 1])

            with left:
                st.markdown("### Explanation")
                st.info(result["action_hint"])

                st.markdown("**Top Rule**")
                st.code(result["top_rule"])

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
    st.subheader("Batch CSV Analysis")
    st.markdown('<div class="small-note">Upload a CSV file with a <code>comment</code> column. Optional columns such as <code>review_id</code>, <code>score</code>, and <code>reference_label</code> can also be included.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

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

            st.markdown("### Overview")
            summary_map = dict(zip(summary_df["metric"], summary_df["value"]))

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Negative", f"{summary_map.get('negative_proportion', 0)*100:.1f}%")
            k2.metric("Neutral", f"{summary_map.get('neutral_proportion', 0)*100:.1f}%")
            k3.metric("Positive", f"{summary_map.get('positive_proportion', 0)*100:.1f}%")
            k4.metric("Match Rate", f"{summary_map.get('overall_match_rate', 0)*100:.1f}%")

            with st.expander("View Summary Table", expanded=False):
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            st.markdown("### Detailed Results")

            with st.expander("Evidence Summary", expanded=True):
                st.dataframe(evidence_summary_df, use_container_width=True, hide_index=True)

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
            st.markdown(f'<div class="download-box">Results were saved under: <code>{output_dir}</code></div>', unsafe_allow_html=True)

            with open(excel_file, "rb") as f:
                st.download_button(
                    label="Download Excel Report",
                    data=f,
                    file_name=f"{dataset_name}_comment_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

with tab3:
    st.subheader("Aspect Configuration")
    st.markdown('<div class="small-note">The aspect dictionary is defined in <code>src/aspect_mapper.py</code>. You can extend it to support new domains.</div>', unsafe_allow_html=True)

    pretty_config = {k: sorted(list(v)) for k, v in ASPECT_CONFIG.items()}

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("### Aspect Dictionary")
        st.json(pretty_config)

    with c2:
        st.markdown("### Copyable JSON")
        st.code(json.dumps(pretty_config, indent=2), language="json")