# HelpDesk AI

An explainable, offline-first IT ticket triage assistant built for an entry-level IT/cybersecurity portfolio.

HelpDesk AI classifies support tickets, estimates priority, retrieves relevant troubleshooting procedures, flags possible security incidents, and stores technician-reviewed results in SQLite. It does **not** automatically execute commands or send responses.

## Portfolio skills demonstrated

- NLP text classification with TF-IDF and logistic regression
- Retrieval over a curated troubleshooting knowledge base
- Confidence scores and human-in-the-loop review
- Security escalation and sensitive-data redaction
- SQLite persistence and audit history
- Streamlit interface and automated tests

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run creates `helpdesk.db` locally. No ticket data leaves the computer.

### Updating an earlier copy

Stop Streamlit with `Ctrl+C`, replace the project files, and launch it again. Streamlit
caches the trained model while running, so a restart is required after model changes.

## Test it

```bash
pytest -q
```

Try these example tickets:

- `I can connect to Wi-Fi but no websites load.`
- `My account is locked after several password attempts.`
- `I clicked a link in an email and entered my Microsoft password.`
- `The shared Finance folder says access denied.`

## How it works

1. A supervised classifier predicts one of six ticket categories.
2. Rule-based controls identify urgent security and outage language.
3. TF-IDF cosine similarity retrieves relevant knowledge articles.
4. The technician reviews, edits, and approves the recommendation.
5. The reviewed outcome is stored for auditing and future evaluation.

## Responsible-AI controls

- Low-confidence predictions are clearly marked for manual review.
- Displayed percentages are model probabilities, not guarantees of correctness.
- Security indicators override ordinary ticket routing.
- Common secrets and sensitive values are redacted before storage.
- Recommendations cite their source articles.
- The system proposes steps but never performs administrative actions.

## Roadmap

- Replace the starter dataset with anonymized lab-generated tickets.
- Add precision, recall, F1, and confusion-matrix evaluation.
- Add Active Directory, DNS, and DHCP lab articles.
- Add optional local-LLM summaries through Ollama.
- Add role-based technician and administrator views.

## Repository structure

```text
app.py                 Streamlit interface
helpdesk_ai.py         Classification, retrieval, safety, and persistence
data/training.csv      Starter labeled ticket dataset
data/knowledge_base.json  Cited troubleshooting procedures
tests/test_helpdesk.py Core behavior tests
```

## Important limitation

The included dataset is deliberately small and synthetic. This is a portfolio MVP, not a production help-desk replacement. Predictions must be reviewed by a person.
