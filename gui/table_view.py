import re
import tkinter as tk
from tkinter import ttk


def _sort_key(value):
    """Sort numbers numerically, text naturally (d2 before d10)."""
    text = str(value)
    try:
        return (0, float(text), [])
    except ValueError:
        parts = [int(p) if p.isdigit() else p.lower()
                 for p in re.split(r"(\d+)", text)]
        return (1, 0.0, parts)


class TableView(ttk.Frame):
    """A Treeview with vertical + horizontal scrollbars and click-to-sort headers."""

    def __init__(self, parent, height=12):
        super().__init__(parent)
        self.columns = []
        self._sort_reverse = {}

        self.tree = ttk.Treeview(self, show="headings", height=height)
        vsb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.tag_configure("odd", background="#f4f7fb")

    def clear(self):
        self.tree.delete(*self.tree.get_children())

    def set_data(self, columns, rows, widths=None):
        """columns: list of header names; rows: list of row lists/tuples."""
        self.columns = list(columns)
        self._sort_reverse = {}
        self.tree.configure(columns=self.columns)

        for i, col in enumerate(self.columns):
            width = (widths[i] if widths and i < len(widths) else 90)
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=width, minwidth=50,
                             anchor="w" if i == 0 else "center")

        self.clear()
        for n, row in enumerate(rows):
            tag = ("odd",) if n % 2 else ()
            self.tree.insert("", "end", values=list(row), tags=tag)

    def _sort_by(self, col):
        idx = self.columns.index(col)
        reverse = self._sort_reverse.get(col, False)
        items = [(self.tree.set(i, col), i) for i in self.tree.get_children("")]
        items.sort(key=lambda x: _sort_key(x[0]), reverse=reverse)
        for pos, (_, iid) in enumerate(items):
            self.tree.move(iid, "", pos)
            self.tree.item(iid, tags=("odd",) if pos % 2 else ())
        self._sort_reverse[col] = not reverse
