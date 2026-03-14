import sys

from PyQt5.QtWidgets import QApplication

from src.viewer import EEGViewer


def main():
    app = QApplication(sys.argv)
    w = EEGViewer()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
