from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).parent
SECURITY_TERMS = {
    "phishing", "malware", "ransomware", "stolen", "unknown login",
    "entered my password", "entered my microsoft password", "suspicious email",
    "suspicious attachment", "encrypted files",
    "data breach", "compromised",
}
OUTAGE_TERMS = {"everyone", "entire office", "all users", "company-wide", "outage"}


@dataclass
class Analysis:
    category: str
    confidence: float
    priority: str
    needs_review: bool
    security_flag: bool
    redacted_text: str
    probabilities: dict[str, float]
    articles: list[dict]


def redact_sensitive(text: str) -> str:
    patterns = [
        (r"(?i)(password|passwd|pwd)\s*[:=]\s*\S+", r"\1: [REDACTED]"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]"),
        (r"\b(?:\d[ -]*?){13,16}\b", "[REDACTED-CARD]"),
        (r"(?i)(api[_ -]?key|token)\s*[:=]\s*\S+", r"\1: [REDACTED]"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    return text


class HelpDeskEngine:
    def __init__(self, training_path: Path | None = None, kb_path: Path | None = None):
        training_path = training_path or ROOT / "data" / "training.csv"
        kb_path = kb_path or ROOT / "data" / "knowledge_base.json"
        training = pd.read_csv(training_path)
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2), stop_words="english", sublinear_tf=True
            )),
            # Mildly reduced regularization helps this deliberately small starter
            # dataset learn its category vocabulary without hiding uncertainty.
            ("classifier", LogisticRegression(max_iter=1000, C=3.0, random_state=42)),
        ])
        self.model.fit(training["text"], training["category"])

        self.articles = json.loads(kb_path.read_text(encoding="utf-8"))
        self.retriever = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        corpus = [f"{a['title']} {a['category']} {a['symptoms']}" for a in self.articles]
        self.article_vectors = self.retriever.fit_transform(corpus)

    def analyze(self, text: str, top_k: int = 2) -> Analysis:
        if not text or not text.strip():
            raise ValueError("Ticket description cannot be empty.")

        redacted = redact_sensitive(text.strip())
        lowered = redacted.lower()
        probabilities = self.model.predict_proba([redacted])[0]
        classes = self.model.classes_
        ranked = sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
        category, confidence = ranked[0]

        security_flag = any(term in lowered for term in SECURITY_TERMS)
        if security_flag:
            category = "Security Incident"
            confidence = max(confidence, 0.99)

        if security_flag:
            priority = "Critical"
        elif any(term in lowered for term in OUTAGE_TERMS):
            priority = "High"
        elif any(term in lowered for term in ("cannot work", "blocked", "urgent")):
            priority = "Medium"
        else:
            priority = "Low"

        query = self.retriever.transform([f"{redacted} {category}"])
        scores = cosine_similarity(query, self.article_vectors)[0]
        indices = scores.argsort()[::-1][:top_k]
        articles = []
        for index in indices:
            article = dict(self.articles[index])
            article["similarity"] = float(scores[index])
            articles.append(article)

        probability_map = {label: float(value) for label, value in ranked}
        return Analysis(
            category=category,
            confidence=float(confidence),
            priority=priority,
            needs_review=confidence < 0.55,
            security_flag=security_flag,
            redacted_text=redacted,
            probabilities=probability_map,
            articles=articles,
        )


def initialize_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                description TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                technician_notes TEXT NOT NULL
            )
        """)


def save_ticket(db_path: Path, analysis: Analysis, status: str, technician_notes: str) -> int:
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """INSERT INTO tickets
            (created_at, description, predicted_category, confidence, priority, status, technician_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                analysis.redacted_text,
                analysis.category,
                analysis.confidence,
                analysis.priority,
                status,
                technician_notes.strip(),
            ),
        )
        return int(cursor.lastrowid)


def load_tickets(db_path: Path) -> pd.DataFrame:
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", connection)
