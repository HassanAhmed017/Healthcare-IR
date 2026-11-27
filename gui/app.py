import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src import pipeline
from src.file_loader import natural_key
from src.positional_index import index_to_rows
from src.query_engine import run_query
from src.retrieval import matrix_to_rows

from gui.table_view import TableView


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR Healthcare - Positional Index & Ranked Retrieval")
        self.geometry("1150x720")
        self.minsize(900, 600)

        self.data = None          # state dict returned by pipeline.build_all
        self.spark = None
        self._queue = queue.Queue()

        self._style()
        self._build_ui()
        self.after(100, self._poll_queue)
        self.load_default()

    # ------------------------------------------------------------------ UI
    def _style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))
        style.configure("Title.TLabel", font=("Segoe UI", 15, "bold"))
        style.configure("Status.TLabel", foreground="#555555")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        # ---- top bar
        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="IR Healthcare Search", style="Title.TLabel").pack(side="left")

        self.btn_import = ttk.Button(top, text="Import folder...", command=self.import_folder)
        self.btn_import.pack(side="right", padx=(6, 0))
        self.btn_default = ttk.Button(top, text="Load default data", command=self.load_default)
        self.btn_default.pack(side="right")

        self.status = tk.StringVar(value="Starting...")
        ttk.Label(self, textvariable=self.status, style="Status.TLabel",
                  padding=(12, 0, 12, 6)).pack(fill="x")

        # ---- tabs
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._build_query_tab()

        self.tab_tables = {}
        for key, title in [
            ("index", "Positional Index"),
            ("tf", "TF Matrix"),
            ("dfidf", "DF / IDF"),
            ("tfidf", "TF-IDF"),
            ("norm", "Normalized TF-IDF"),
            ("len", "Doc Lengths"),
        ]:
            table = TableView(self.tabs)
            self.tabs.add(table, text=title)
            self.tab_tables[key] = table

    def _build_query_tab(self):
        tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(tab, text="Search")

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Label(bar, text="Query:").pack(side="left")
        self.query_var = tk.StringVar()
        entry = ttk.Entry(bar, textvariable=self.query_var, font=("Segoe UI", 11))
        entry.pack(side="left", fill="x", expand=True, padx=8)
        entry.bind("<Return>", lambda e: self.search())
        entry.focus_set()
        self.btn_search = ttk.Button(bar, text="Search", style="Accent.TButton",
                                     command=self.search)
        self.btn_search.pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear_query).pack(side="left", padx=(6, 0))

        ttk.Label(
            tab, foreground="#666666",
            text="Examples:  patient care   |   patient AND safety   |   medical AND NOT research",
        ).pack(anchor="w", pady=(4, 8))

        panes = ttk.PanedWindow(tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        # left: text details
        left = ttk.Frame(panes)
        panes.add(left, weight=1)
        ttk.Label(left, text="Details", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        text_frame = ttk.Frame(left)
        text_frame.pack(fill="both", expand=True)
        self.details = tk.Text(text_frame, wrap="word", font=("Consolas", 10),
                               state="disabled", relief="solid", borderwidth=1)
        sb = ttk.Scrollbar(text_frame, command=self.details.yview)
        self.details.configure(yscrollcommand=sb.set)
        self.details.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # right: ranking + query vector
        right = ttk.Frame(panes)
        panes.add(right, weight=1)
        ttk.Label(right, text="Ranking", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.rank_table = TableView(right, height=10)
        self.rank_table.pack(fill="both", expand=True)
        ttk.Label(right, text="Query vector", font=("Segoe UI", 10, "bold")
                  ).pack(anchor="w", pady=(8, 0))
        self.qvec_table = TableView(right, height=6)
        self.qvec_table.pack(fill="both", expand=True)

    # ------------------------------------------------------- async helpers
    def _run_async(self, work, on_done, message):
        self._set_busy(True, message)

        def target():
            try:
                self._queue.put((on_done, work(), None))
            except Exception as error:  # noqa: BLE001
                self._queue.put((on_done, None, error))

        threading.Thread(target=target, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                callback, result, error = self._queue.get_nowait()
                callback(result, error)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _set_busy(self, busy, message=None):
        state = "disabled" if busy else "normal"
        for btn in (self.btn_import, self.btn_default, self.btn_search):
            btn.configure(state=state)
        self.configure(cursor="watch" if busy else "")
        if message:
            self.status.set(message)

    def _get_spark(self):
        if self.spark is None:
            self.spark = pipeline.spark_get()
        return self.spark

    # ------------------------------------------------------------- loading
    def load_default(self):
        self._run_async(lambda: pipeline.load_default(self._get_spark()),
                        self._on_loaded, "Building index with Spark (first run can take a few seconds)...")

    def import_folder(self):
        folder = filedialog.askdirectory(title="Choose a folder with documents")
        if not folder:
            return
        self._run_async(lambda: pipeline.import_dataset(self._get_spark(), folder),
                        self._on_loaded, f"Importing {folder} ...")

    def _on_loaded(self, state, error):
        self._set_busy(False)
        if error:
            self.status.set("Load failed.")
            messagebox.showerror("Error", str(error))
            return

        self.data = state
        self._fill_tables()
        self.clear_query()
        self.status.set(
            f"Loaded {state['N']} documents, {len(state['index'])} terms  |  "
            f"Folder: {state['folder']}  |  Output saved to /output"
        )
        if state.get("warnings"):
            messagebox.showwarning("Warnings", "\n".join(state["warnings"]))

    def _fill_tables(self):
        s = self.data
        doc_ids = s["doc_ids"]

        # positional index
        rows = index_to_rows(s["index"])
        self.tab_tables["index"].set_data(
            ["Term", "Postings (doc: positions)", "DF"],
            [(t, p, df) for t, p, df in rows], widths=[140, 700, 60])

        # TF matrix (raw + weighted per doc)
        header = ["Term"]
        for d in doc_ids:
            header += [f"{d} TF", f"{d} W-TF"]
        tf_rows = []
        for term in sorted(s["raw_tf"]):
            row = [term]
            for d in doc_ids:
                row.append(s["raw_tf"][term].get(d, 0))
                row.append(round(s["weighted_tf"].get(term, {}).get(d, 0), 4))
            tf_rows.append(row)
        self.tab_tables["tf"].set_data(header, tf_rows, widths=[120] + [70] * (len(header) - 1))

        # DF / IDF
        self.tab_tables["dfidf"].set_data(
            ["Term", "DF", "IDF"],
            [[t, s["df"][t], round(s["idf"][t], 4)] for t in sorted(s["df"])],
            widths=[160, 80, 100])

        # TF-IDF and normalized
        mh = ["Term"] + list(doc_ids)
        w = [120] + [70] * len(doc_ids)
        self.tab_tables["tfidf"].set_data(mh, matrix_to_rows(s["tfidf"], doc_ids), widths=w)
        self.tab_tables["norm"].set_data(mh, matrix_to_rows(s["normalized"], doc_ids), widths=w)

        # doc lengths
        self.tab_tables["len"].set_data(
            ["Doc", "Sum of squares", "Length"],
            [[d, round(s["lengths"][d][0], 4), round(s["lengths"][d][1], 4)] for d in doc_ids],
            widths=[100, 140, 140])

    # --------------------------------------------------------------- query
    def clear_query(self):
        self.query_var.set("")
        self._set_details("")
        self.rank_table.set_data(["Rank", "Doc", "Score", "Preview"], [], widths=[50, 70, 80, 300])
        self.qvec_table.set_data(["Term", "TF", "W-TF", "IDF", "TF-IDF", "Norm"], [])

    def _set_details(self, text):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def search(self):
        if not self.data:
            messagebox.showinfo("Not ready", "Data is still loading.")
            return
        text = self.query_var.get().strip()
        if not text:
            self._set_details("Please type a query.")
            return

        result = run_query(text, self.data)
        try:
            pipeline.save_query_result(text, result)
        except OSError:
            pass
        self._show_result(text, result)

    def _show_result(self, text, result):
        lines = [f"Query: {text}", f"Type:  {result['type']}"]
        if result["message"]:
            lines.append(f"Note:  {result['message']}")

        lines += ["", "Phrase matches:"]
        for m in result["matches"]:
            if not m["docs"]:
                lines.append(f"  '{m['phrase']}': no match")
                continue
            for doc, starts in m["docs"].items():
                lines.append(f"  '{m['phrase']}' in {doc}: starts {starts}, "
                             f"tokens {m['ranges'][doc]}")

        lines += ["", "Matched documents: " + (", ".join(result["matched_docs"]) or "none")]

        if result["details"]:
            lines += ["", "Similarity (query x doc):"]
            for doc, steps in result["details"].items():
                calc = " + ".join(f"{q:.4f}x{d:.4f}" for _, q, d, _ in steps) or "0"
                lines.append(f"  {doc}: {calc} = {result['scores'][doc]:.4f}")

        self._set_details("\n".join(lines))

        # ranking table
        rank_rows = []
        for i, (doc, score) in enumerate(result["ranking"], start=1):
            preview = " ".join(self.data["docs"][doc][:10])
            rank_rows.append([i, doc, f"{score:.4f}", preview])
        self.rank_table.set_data(["Rank", "Doc", "Score", "Preview"], rank_rows,
                                 widths=[50, 70, 80, 300])

        # query vector table
        qv = result["query_vector"]
        qrows = [[r["term"], r["tf"], round(r["w_tf"], 4), round(r["idf"], 4),
                  round(r["tfidf"], 4), round(r["normalized"], 4)] for r in qv["rows"]]
        self.qvec_table.set_data(["Term", "TF", "W-TF", "IDF", "TF-IDF", "Norm"], qrows,
                                 widths=[110, 40, 70, 70, 70, 70])
