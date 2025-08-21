import sys
from PyQt5.QtWidgets import QApplication

def main():
    try:
        from meshit_workflow_gui import MeshItWorkflowGUI
    except Exception as e:
        print("Failed to import GUI module:", e, file=sys.stderr)
        raise

    app = QApplication(sys.argv)
    win = MeshItWorkflowGUI()
    win.show()
    # use exec() (preferred) but exec_() also works for PyQt5
    sys.exit(app.exec())

if __name__ == "__main__":
    main()