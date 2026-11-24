import re


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(value))
    ]


def build_positional_index(spark, docs):
    sc = spark.sparkContext
    rdd = sc.parallelize(list(docs.items()))

    term_positions = rdd.flatMap(
        lambda doc: [
            ((word, doc[0]), position)
            for position, word in enumerate(doc[1], start=1)
        ]
    )

    grouped_positions = term_positions.groupByKey().mapValues(
        lambda positions: sorted(list(positions))
    )

    term_documents = grouped_positions.map(
        lambda item: (item[0][0], (item[0][1], item[1]))
    )

    grouped_documents = term_documents.groupByKey()

    collected = grouped_documents.collect()

    index = {}

    for term, postings in sorted(collected, key=lambda item: item[0]):
        index[term] = {}

        for doc_id, positions in sorted(
                postings, key=lambda item: natural_key(item[0])
        ):
            index[term][doc_id] = positions

    return index


def format_postings(postings):
    parts = []

    for doc_id, positions in sorted(
            postings.items(), key=lambda item: natural_key(item[0])
    ):
        position_text = ", ".join(str(pos) for pos in positions)
        parts.append(f"{doc_id}: {position_text}")

    return "; ".join(parts)


def index_to_rows(index):
    rows = []

    for term, postings in index.items():
        postings_text = format_postings(postings)
        df = len(postings)

        rows.append((term, postings_text, df))

    return rows