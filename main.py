import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from src.viewer import EEGViewer


def main():
    Path("logs").mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    w = EEGViewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
