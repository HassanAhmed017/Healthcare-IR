import csv
import os
import shutil
import sys
import threading
from pathlib import Path

from src.file_loader import load_folder, natural_key
from src.positional_index import build_positional_index
from src.retrieval import build_all_matrices, matrix_to_rows

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "data" / "default"
IMPORTED_DIR = ROOT / "data" / "imported"
TEMP_DIR = ROOT / "data" / "_importing"
BACKUP_DIR = ROOT / "data" / "_backup"
OUTPUT_DIR = ROOT / "output"
QUERY_LOG = OUTPUT_DIR / "query_results.txt"
SUPPORTED = {".txt", ".docx", ".xlsx", ".pdf"}

_spark = None
_spark_lock = threading.Lock()


def spark_get():
    global _spark
    with _spark_lock:
        if _spark is None:
            os.environ["PYSPARK_PYTHON"] = sys.executable
            os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
            from pyspark.sql import SparkSession
            _spark = (
                SparkSession.builder
                .master("local[*]")
                .appName("IR-Healthcare")
                .config("spark.ui.enabled", "false")
                .getOrCreate()
            )
            _spark.sparkContext.setLogLevel("ERROR")
        return _spark


def build_all(spark, folder):
    docs, warnings = load_folder(folder)
    doc_ids = sorted(docs, key=natural_key)
    index = build_positional_index(spark, docs)
    state = build_all_matrices(index, doc_ids)
    state.update({
        "docs": docs,
        "doc_ids": doc_ids,
        "index": index,
        "warnings": warnings,
        "folder": str(folder),
    })
    return state


def load_default(spark):
    if not DEFAULT_DIR.is_dir():
        raise ValueError(f"Default data folder is missing: {DEFAULT_DIR}")
    state = build_all(spark, DEFAULT_DIR)
    save_outputs(state)
    return state


def _supported_files(folder):
    return [p for p in Path(folder).iterdir()
            if p.is_file()
            and p.suffix.lower() in SUPPORTED
            and not p.name.startswith("~$")]


def import_dataset(spark, src_folder):
    src = Path(src_folder).resolve()
    if not src.is_dir():
        raise ValueError(f"Folder does not exist: {src}")

    protected = {d.resolve() for d in (DEFAULT_DIR, IMPORTED_DIR, TEMP_DIR, BACKUP_DIR)}
    if src in protected:
        raise ValueError("Choose a folder other than the project data folders")

    files = _supported_files(src)
    if not files:
        raise ValueError("The folder has no supported files (txt, docx, xlsx, pdf)")

    state = build_all(spark, src)

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)
    try:
        for p in files:
            shutil.copy2(p, TEMP_DIR / p.name)

        if BACKUP_DIR.exists():
            shutil.rmtree(BACKUP_DIR)
        if IMPORTED_DIR.exists():
            IMPORTED_DIR.rename(BACKUP_DIR)
        try:
            TEMP_DIR.rename(IMPORTED_DIR)
        except Exception:
            if BACKUP_DIR.exists():
                BACKUP_DIR.rename(IMPORTED_DIR)
            raise
        shutil.rmtree(BACKUP_DIR, ignore_errors=True)
    except Exception:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        raise

    state["folder"] = str(IMPORTED_DIR)
    save_outputs(state)
    return state


def _write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _write_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(content)


def save_outputs(state, out_dir=OUTPUT_DIR):
    doc_ids = state["doc_ids"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_lines = []
    for term in sorted(state["index"]):
        postings = state["index"][term]
        details = " | ".join(
            f"{doc}: {', '.join(map(str, postings[doc]))}"
            for doc in sorted(postings, key=natural_key)
        )
        index_lines.append(f"{term} -> {details}")
    _write_text(out / "positional_index.txt", "\n".join(index_lines) + "\n")

    tf_header = ["Term"]
    for doc in doc_ids:
        tf_header += [f"{doc}_Raw_TF", f"{doc}_Weighted_TF"]

    tf_rows = []
    for term in sorted(state["raw_tf"]):
        row = [term]
        for doc in doc_ids:
            row.append(state["raw_tf"][term].get(doc, 0))
            row.append(round(state["weighted_tf"].get(term, {}).get(doc, 0), 4))
        tf_rows.append(row)
    _write_csv(out / "tf_matrix.csv", tf_header, tf_rows)

    _write_csv(
        out / "df_idf.csv",
        ["Term", "DF", "IDF"],
        [[t, state["df"][t], round(state["idf"][t], 4)] for t in sorted(state["df"])],
        )

    matrix_header = ["Term"] + list(doc_ids)
    _write_csv(out / "tfidf_matrix.csv", matrix_header,
               matrix_to_rows(state["tfidf"], doc_ids))

    _write_csv(
        out / "doc_lengths.csv",
        ["Doc", "SumSquares", "Length"],
        [[d, round(state["lengths"][d][0], 4), round(state["lengths"][d][1], 4)]
         for d in doc_ids],
        )

    _write_csv(out / "normalized_tfidf.csv", matrix_header,
               matrix_to_rows(state["normalized"], doc_ids))


def format_query_result(query_text, result):
    lines = [f"Query: {query_text}", f"Type: {result['type']}"]
    if result["message"]:
        lines.append(f"Note: {result['message']}")

    lines += ["", "Phrase matches:"]
    for m in result["matches"]:
        if not m["docs"]:
            lines.append(f"  '{m['phrase']}': no match")
            continue
        for doc, starts in m["docs"].items():
            lines.append(f"  '{m['phrase']}' in {doc}: starts {starts}, "
                         f"tokens {m['ranges'][doc]}")

    lines += ["", "Matched documents: " + (", ".join(result["matched_docs"]) or "none")]

    qv = result["query_vector"]
    if qv["rows"]:
        lines += ["", "Query vector:",
                  f"  {'term':<15}{'tf':>4}{'w_tf':>9}{'idf':>9}{'tf-idf':>9}{'norm':>9}"]
        for r in qv["rows"]:
            lines.append(f"  {r['term']:<15}{r['tf']:>4}{r['w_tf']:>9.4f}"
                         f"{r['idf']:>9.4f}{r['tfidf']:>9.4f}{r['normalized']:>9.4f}")
        lines.append(f"  length = {qv['length']:.4f}")

    if result["details"]:
        lines += ["", "Similarity:"]
        for doc, steps in result["details"].items():
            calc = " + ".join(f"{q:.4f}x{d:.4f}" for _, q, d, _ in steps) or "0"
            lines.append(f"  {doc}: {calc} = {result['scores'][doc]:.4f}")

    if result["ranking"]:
        lines += ["", "Ranking:"]
        for i, (doc, score) in enumerate(result["ranking"], start=1):
            lines.append(f"  {i}. {doc}  {score:.4f}")

    return "\n".join(lines)


def save_query_result(query_text, result, path=QUERY_LOG):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(format_query_result(query_text, result) + "\n" + "-" * 60 + "\n")
