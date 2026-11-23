import re
from pathlib import Path

from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".txt", ".docx", ".xlsx", ".pdf"}


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def natural_key(doc_id):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(doc_id))
    ]


def read_txt(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="cp1252")


def read_docx(path):
    document = Document(path)
    parts = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    parts.append(cell.text)

    return "\n".join(parts)


def read_xlsx(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    parts = []

    try:
        for sheet in workbook.worksheets:
            for row in sheet.iter_rows(values_only=True):
                for value in row:
                    if value is not None and str(value).strip():
                        parts.append(str(value))
    finally:
        workbook.close()

    return " ".join(parts)


def read_pdf(path):
    reader = PdfReader(path)
    parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)

    return "\n".join(parts)


def read_file(path):
    path = Path(path)
    extension = path.suffix.lower()

    if extension == ".txt":
        return read_txt(path)
    elif extension == ".docx":
        return read_docx(path)
    elif extension == ".xlsx":
        return read_xlsx(path)
    elif extension == ".pdf":
        return read_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_folder(folder):
    folder = Path(folder)

    if not folder.is_dir():
        raise ValueError(f"Folder does not exist: {folder}")

    docs = {}
    warnings = []

    files = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and not path.name.startswith("~$")
        ),
        key=lambda path: natural_key(path.name)
    )

    if not files:
        raise ValueError("No supported documents found in the folder.")

    # Detect filenames that share the same stem
    stem_counts = {}

    for path in files:
        stem = path.stem.lower()
        stem_counts[stem] = stem_counts.get(stem, 0) + 1

    for path in files:
        stem = path.stem.lower()

        if stem_counts[stem] > 1:
            doc_id = path.name
        else:
            doc_id = path.stem

        try:
            text = read_file(path)
            tokens = tokenize(text)

            if not tokens:
                warnings.append(
                    f"Skipped empty or unreadable document: {path.name}"
                )
                continue

            docs[doc_id] = tokens

        except Exception as error:
            warnings.append(
                f"Could not load {path.name}: {error}"
            )

    if not docs:
        raise ValueError(
            "No readable documents were found. "
            + " ".join(warnings)
        )

    docs = dict(
        sorted(docs.items(), key=lambda item: natural_key(item[0]))
    )

    return docs, warnings
