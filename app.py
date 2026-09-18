from pathlib import Path

import pandas as pd
import streamlit as st

from helpdesk_ai import HelpDeskEngine, load_tickets, save_ticket


st.set_page_config(page_title="HelpDesk AI", page_icon="🛠️", layout="wide")
DB_PATH = Path(__file__).parent / "helpdesk.db"


@st.cache_resource
def get_engine() -> HelpDeskEngine:
    return HelpDeskEngine()


engine = get_engine()
st.title("HelpDesk AI")
st.caption("Explainable IT ticket triage with human review")

triage_tab, history_tab, about_tab = st.tabs(["Triage Ticket", "Ticket History", "About the Model"])

with triage_tab:
    description = st.text_area(
        "Describe the issue",
        height=150,
        placeholder="Example: I clicked a link in an email and entered my Microsoft password.",
    )
    if st.button("Analyze ticket", type="primary", use_container_width=True):
        try:
            st.session_state.analysis = engine.analyze(description)
        except ValueError as error:
            st.error(str(error))

    analysis = st.session_state.get("analysis")
    if analysis:
        col1, col2, col3 = st.columns(3)
        col1.metric("Category", analysis.category)
        col2.metric("Priority", analysis.priority)
        col3.metric("Confidence", f"{analysis.confidence:.0%}")

        if analysis.security_flag:
            st.error("Potential security incident: stop interaction and escalate through the approved incident process.")
        elif analysis.needs_review:
            st.warning("Low-confidence prediction: technician review is required.")

        if analysis.redacted_text != description.strip():
            st.info("Sensitive-looking values were redacted before storage.")

        st.subheader("Recommended knowledge articles")
        for article in analysis.articles:
            with st.expander(f"{article['id']} · {article['title']} · match {article['similarity']:.0%}", expanded=True):
                for number, step in enumerate(article["steps"], start=1):
                    st.write(f"{number}. {step}")
                st.warning(article["warning"])

        with st.expander("Model explanation"):
            chart_data = pd.DataFrame(
                {"category": list(analysis.probabilities), "probability": list(analysis.probabilities.values())}
            ).set_index("category")
            st.bar_chart(chart_data)
            st.caption("The classifier uses word and two-word TF-IDF features with logistic regression.")

        technician_notes = st.text_area("Technician notes", placeholder="Record validation, changes, and outcome.")
        status = st.selectbox("Review decision", ["Reviewed - pending work", "Resolved", "Escalated", "Prediction corrected"])
        if st.button("Save reviewed ticket", use_container_width=True):
            ticket_id = save_ticket(DB_PATH, analysis, status, technician_notes)
            st.success(f"Ticket #{ticket_id} saved.")

with history_tab:
    history = load_tickets(DB_PATH)
    if history.empty:
        st.info("No reviewed tickets have been saved yet.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Export ticket history as CSV",
            history.to_csv(index=False),
            file_name="helpdesk_ticket_history.csv",
            mime="text/csv",
        )

with about_tab:
    st.subheader("What this MVP demonstrates")
    st.markdown(
        """
        - Supervised NLP classification for ticket routing
        - Retrieval of cited troubleshooting procedures
        - Confidence reporting instead of pretending every prediction is correct
        - Security overrides, redaction, and human approval
        - Local SQLite audit history

        **Limitations:** The starter training data is small and synthetic. This system must not be used
        as an autonomous production help desk or a substitute for an organization's security procedures.
        """
    )
