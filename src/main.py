import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gui.app import App

def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
