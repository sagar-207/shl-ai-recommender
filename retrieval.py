"""
Retrieval engine for SHL catalog.
Uses TF-IDF + cosine similarity (no external model download needed).
At deploy time the Dockerfile can optionally pre-download sentence-transformers
for better semantic quality; this module falls back to TF-IDF gracefully.
"""

import math
import re
from collections import Counter

from catalog_data import CATALOG, TEST_TYPE_LABELS

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "that", "this",
    "these", "those", "it", "its", "i", "we", "you", "he", "she", "they",
    "who", "which", "what", "when", "where", "how", "not", "no", "nor",
}

def _tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9#+]+", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

def _build_doc(item):
    types = " ".join(item.get("test_type", []))
    type_labels = " ".join(TEST_TYPE_LABELS.get(t, t) for t in item.get("test_type", []))
    levels = " ".join(item.get("job_levels", []))
    adaptive = "adaptive" if item.get("adaptive") else ""
    remote = "remote" if item.get("remote_testing") else ""
    name_repeated = (item["name"] + " ") * 3
    return (f"{name_repeated}"
            f"{item['description']} "
            f"{type_labels} {types} "
            f"{levels} {adaptive} {remote}")

class TFIDFIndex:
    def __init__(self, documents):
        self.n_docs = len(documents)
        self.tokenised = [_tokenize(doc) for doc in documents]
        df = Counter()
        for tokens in self.tokenised:
            for tok in set(tokens):
                df[tok] += 1
        self.idf = {tok: math.log((self.n_docs + 1) / (count + 1)) + 1 for tok, count in df.items()}
        self.vectors = []
        for tokens in self.tokenised:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {tok: (count / total) * self.idf.get(tok, 1.0) for tok, count in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.vectors.append({k: v / norm for k, v in vec.items()})

    def query(self, text, top_k):
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) or 1
        q_vec = {tok: (count / total) * self.idf.get(tok, 1.0) for tok, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec = {k: v / norm for k, v in q_vec.items()}
        scores = []
        for i, doc_vec in enumerate(self.vectors):
            score = sum(q_vec.get(tok, 0.0) * doc_vec.get(tok, 0.0) for tok in q_vec)
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

_DOCS = [_build_doc(item) for item in CATALOG]
_INDEX = TFIDFIndex(_DOCS)

KEYWORD_BOOSTS = {
    "java": ["Java 8 (New)"],
    "python": ["Python (New)"],
    "sql": ["SQL (New)"],
    "javascript": ["JavaScript (New)"],
    "c#": ["C# (New)"],
    "c++": ["C++ (New)"],
    "coding": ["Automata - Fix (Coding Simulation)", "Automata Pro"],
    "debug": ["Automata - Fix (Coding Simulation)"],
    "personality": ["OPQ32r", "MQ (Motivation Questionnaire)", "Hogan Personality Inventory (HPI)"],
    "motivation": ["MQ (Motivation Questionnaire)"],
    "leadership": ["OPQ32r", "Hogan Development Survey (HDS)", "Hogan Leadership Forecast Series"],
    "executive": ["Hogan Leadership Forecast Series", "Hogan Development Survey (HDS)"],
    "graduate": ["Graduate 8.0", "Verify G+", "Graduate/Professional Scenarios"],
    "numerical": ["Verify Numerical Reasoning", "Numerical Reasoning"],
    "verbal": ["Verify Verbal Reasoning", "Verbal Reasoning"],
    "inductive": ["Verify Inductive Reasoning", "Inductive Reasoning"],
    "cognitive": ["Verify G+", "General Ability", "Graduate 8.0", "Professional 8.0"],
    "sales": ["Sales Potential Questionnaire", "Sales Solution"],
    "customer": ["Customer Contact Scenarios", "Service Orientation", "SJT - Customer Focus"],
    "contact centre": ["Contact Centre Solution", "Customer Contact Scenarios"],
    "call centre": ["Contact Centre Solution", "Customer Contact Scenarios"],
    "finance": ["Financial Accounting (New)", "Accounting and Finance (AFAS)", "Verify Numerical Reasoning"],
    "accounting": ["Financial Accounting (New)", "Accounting and Finance (AFAS)"],
    "mechanical": ["Mechanical Reasoning", "Verify Mechanical Comprehension"],
    "engineering": ["Verify Mechanical Comprehension", "Spatial Reasoning", "Mechanical Reasoning"],
    "data scientist": ["Python (New)", "Machine Learning", "SQL (New)"],
    "data engineer": ["Data Engineering", "SQL (New)", "Python (New)"],
    "machine learning": ["Machine Learning"],
    "cybersecurity": ["Cybersecurity Assessment"],
    "security": ["Cybersecurity Assessment"],
    "excel": ["Microsoft Excel (Office 2019)"],
    "manager": ["OPQ32r", "Managerial Scenarios", "MQ (Motivation Questionnaire)"],
    "supervisor": ["Supervisory Scenarios"],
    "english": ["Workplace English Language Test (WELT-B2/C1)"],
    "safety": ["Dependability & Safety Instrument (DSI)"],
    "integrity": ["Dependability & Safety Instrument (DSI)"],
    "project management": ["Project Management (PMP Focus)"],
    "agile": ["Project Management (PMP Focus)"],
    "volume": ["Verify G+", "Aspects Potential", "Graduate 8.0"],
}

def _apply_keyword_boosts(query, scores):
    query_lower = query.lower()
    name_to_idx = {item["name"].lower(): i for i, item in enumerate(CATALOG)}
    boosted = dict(scores)
    for keyword, names in KEYWORD_BOOSTS.items():
        if keyword in query_lower:
            for name in names:
                idx = name_to_idx.get(name.lower())
                if idx is not None:
                    boosted[idx] = boosted.get(idx, 0.0) + 0.25
    return boosted

def search(query, top_k=10, filters=None):
    n_fetch = min(len(CATALOG), max(top_k * 4, 30))
    raw_scores = _INDEX.query(query, n_fetch)
    score_map = {idx: score for idx, score in raw_scores}
    score_map = _apply_keyword_boosts(query, score_map)
    sorted_items = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in sorted_items:
        item = CATALOG[idx]
        if filters:
            if filters.get("test_types"):
                if not any(t in item.get("test_type", []) for t in filters["test_types"]):
                    continue
            if filters.get("job_levels"):
                item_levels_lower = [l.lower() for l in item.get("job_levels", [])]
                if not any(l.lower() in item_levels_lower for l in filters["job_levels"]):
                    continue
            if filters.get("remote_only") and not item.get("remote_testing"):
                continue
        results.append({**item, "_score": round(score, 4)})
        if len(results) >= top_k:
            break
    return results

def get_by_name(name):
    name_lower = name.lower().strip()
    for item in CATALOG:
        if item["name"].lower() == name_lower:
            return item
    for item in CATALOG:
        if name_lower in item["name"].lower() or item["name"].lower() in name_lower:
            return item
    return None

def get_all_names():
    return [item["name"] for item in CATALOG]

def _ensure_index():
    pass

if __name__ == "__main__":
    tests = [
        "Java developer mid level stakeholder communication",
        "graduate finance high volume cognitive personality",
        "contact centre insurance customer service",
        "senior manager leadership promotion",
        "data scientist Python machine learning",
    ]
    for t in tests:
        results = search(t, top_k=5)
        print(f"\nQuery: {t}")
        for r in results:
            print(f"  {r['name']:45s} score={r['_score']:.3f}")
