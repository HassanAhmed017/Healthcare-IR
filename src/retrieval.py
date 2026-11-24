import math


def compute_raw_tf(doc_index):
    tf = {}
    for word, positions in doc_index.items():
        tf[word] = len(positions)
    return tf


def compute_df(all_docs_index):
    df = {}
    for doc_index in all_docs_index.values():
        for word in doc_index:
            df[word] = df.get(word, 0) + 1
    return df


def compute_idf(df, total_documents):
    idf = {}
    for word, frequency in df.items():
        idf[word] = math.log10(total_documents / frequency)
    return idf


def compute_weighted_tf(tf):
    wtf = {}
    for word, frequency in tf.items():
        if frequency > 0:
            wtf[word] = 1 + math.log10(frequency)
    return wtf


def compute_tfidf(wtf, idf):
    tfidf = {}
    for word, weight in wtf.items():
        tfidf[word] = weight * idf[word]
    return tfidf


def compute_doc_lengths(tfidf):
    sum_sq = 0
    for weight in tfidf.values():
        sum_sq += weight ** 2
    return sum_sq, math.sqrt(sum_sq)


def normalize_matrix(tfidf, doc_vector_length):
    normalized = {}
    for word, weight in tfidf.items():
        if doc_vector_length == 0:
            normalized[word] = 0.0
        else:
            normalized[word] = weight / doc_vector_length
    return normalized


def split_index_by_doc(index, doc_ids):
    per_doc = {d: {} for d in doc_ids}
    for term, postings in index.items():
        for doc, positions in postings.items():
            per_doc.setdefault(doc, {})[term] = positions
    return per_doc


def to_term_doc(per_doc_matrix):
    matrix = {}
    for doc, terms in per_doc_matrix.items():
        for term, value in terms.items():
            matrix.setdefault(term, {})[doc] = value
    return matrix


def to_doc_vectors(matrix):
    vectors = {}
    for term, docs in matrix.items():
        for doc, value in docs.items():
            vectors.setdefault(doc, {})[term] = value
    return vectors


def build_all_matrices(index, doc_ids):
    N = len(doc_ids)
    per_doc_index = split_index_by_doc(index, doc_ids)

    df = compute_df(per_doc_index)
    idf = compute_idf(df, N)

    raw_tf, w_tf, tfidf, lengths, normalized = {}, {}, {}, {}, {}
    for doc in doc_ids:
        raw_tf[doc] = compute_raw_tf(per_doc_index[doc])
        w_tf[doc] = compute_weighted_tf(raw_tf[doc])
        tfidf[doc] = compute_tfidf(w_tf[doc], idf)
        lengths[doc] = compute_doc_lengths(tfidf[doc])
        normalized[doc] = normalize_matrix(tfidf[doc], lengths[doc][1])

    return {
        "N": N,
        "raw_tf": to_term_doc(raw_tf),
        "weighted_tf": to_term_doc(w_tf),
        "df": df,
        "idf": idf,
        "tfidf": to_term_doc(tfidf),
        "lengths": lengths,
        "normalized": to_term_doc(normalized),
        "doc_vectors": normalized,
    }


def matrix_to_rows(matrix, doc_ids, decimals=4):
    rows = []
    for term in sorted(matrix):
        row = [term]
        for doc in doc_ids:
            row.append(round(matrix[term].get(doc, 0), decimals))
        rows.append(row)
    return rows