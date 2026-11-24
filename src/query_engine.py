import math
import re
from collections import Counter

from src.file_loader import tokenize, natural_key


def parse_query(text):
    text = text.lower()
    if re.search(r"\band\s+not\b", text):
        qtype, parts = "AND_NOT", re.split(r"\band\s+not\b", text)
    elif re.search(r"\band\b", text):
        qtype, parts = "AND", re.split(r"\band\b", text)
    else:
        qtype, parts = "PHRASE", [text]

    phrases = [toks for toks in (tokenize(p) for p in parts) if toks]
    if len(phrases) < 2:
        qtype = "PHRASE"
    return qtype, (phrases[:1] if qtype == "PHRASE" else phrases[:2])


def phrase_search(index, phrase_tokens):
    if not phrase_tokens or any(t not in index for t in phrase_tokens):
        return {}

    candidate_docs = set(index[phrase_tokens[0]])
    for t in phrase_tokens[1:]:
        candidate_docs &= set(index[t])

    result = {}
    for doc in sorted(candidate_docs, key=natural_key):
        pos = [set(index[t][doc]) for t in phrase_tokens]
        starts = [p for p in sorted(pos[0])
                  if all(p + i in pos[i] for i in range(1, len(phrase_tokens)))]
        if starts:
            result[doc] = starts
    return result


def format_ranges(starts, k):
    parts = []
    for p in sorted(starts):
        parts.append(str(p) if k <= 1 else f"{p}-{p + k - 1}")
    return ", ".join(parts)


def boolean_and(a, b):
    return set(a) & set(b)


def boolean_and_not(a, b):
    return set(a) - set(b)


def select_query_terms(qtype, phrases):
    if not phrases:
        return []
    if qtype == "AND":
        return [w for p in phrases for w in p]
    return list(phrases[0])


def build_query_vector(terms, idf):
    counts = Counter(terms)
    rows, skipped = [], []
    for term, tf in counts.items():
        if term not in idf:
            skipped.append(term)
            continue
        w_tf = 1 + math.log10(tf)
        tfidf = w_tf * idf[term]
        rows.append({"term": term, "tf": tf, "w_tf": w_tf,
                     "idf": idf[term], "tfidf": tfidf, "normalized": 0.0})

    length = math.sqrt(sum(r["tfidf"] ** 2 for r in rows))
    for r in rows:
        r["normalized"] = r["tfidf"] / length if length > 0 else 0.0

    vector = {r["term"]: r["normalized"] for r in rows}
    return {"rows": rows, "length": length, "vector": vector, "skipped": skipped}


def compute_similarity(query_vec, doc_vectors, docs):
    scores, details = {}, {}
    for doc in docs:
        dvec = doc_vectors.get(doc, {})
        steps = []
        for term, q_w in query_vec.items():
            d_w = dvec.get(term, 0.0)
            if d_w > 0:
                steps.append((term, q_w, d_w, q_w * d_w))
        details[doc] = steps
        scores[doc] = sum(s[3] for s in steps)
    return scores, details


def rank_results(scores):
    return sorted(scores.items(),
                  key=lambda x: (-round(x[1], 10), natural_key(x[0])))


def run_query(text, state):
    text = text or ""
    result = {"type": "PHRASE", "phrases": [], "matches": [], "matched_docs": [],
              "terms": [],
              "query_vector": {"rows": [], "length": 0.0, "vector": {}, "skipped": []},
              "scores": {}, "details": {}, "ranking": [], "message": ""}

    qtype, phrases = parse_query(text)
    result["type"], result["phrases"] = qtype, phrases
    if not phrases:
        result["message"] = "Empty query."
        return result

    notes = []
    operators = len(re.findall(r"\band\b", text.lower()))
    if operators > 1:
        notes.append("Only one AND / AND NOT is supported; extra phrases were ignored.")
    elif operators == 1 and qtype == "PHRASE":
        notes.append("Incomplete Boolean query; searched as a single phrase.")

    index = state.get("index", {})
    idf = state.get("idf", {})
    doc_vectors = state.get("doc_vectors", {})

    found = [phrase_search(index, p) for p in phrases]
    result["matches"] = [
        {"phrase": " ".join(p), "docs": f,
         "ranges": {d: format_ranges(s, len(p)) for d, s in f.items()}}
        for p, f in zip(phrases, found)
    ]

    if qtype == "AND":
        matched = boolean_and(found[0], found[1])
    elif qtype == "AND_NOT":
        matched = boolean_and_not(found[0], found[1])
    else:
        matched = set(found[0])
    matched_docs = sorted(matched, key=natural_key)
    result["matched_docs"] = matched_docs

    terms = select_query_terms(qtype, phrases)
    query_vector = build_query_vector(terms, idf)
    result["terms"] = terms
    result["query_vector"] = query_vector
    if query_vector["skipped"]:
        notes.append("Ignored terms not found in the index: "
                     + ", ".join(query_vector["skipped"]) + ".")

    if not matched_docs:
        notes.append("No matching documents.")
        result["message"] = " ".join(notes)
        return result

    scores, details = compute_similarity(query_vector["vector"], doc_vectors, matched_docs)
    result["scores"], result["details"] = scores, details
    result["ranking"] = rank_results(scores)
    result["message"] = " ".join(notes)
    return result