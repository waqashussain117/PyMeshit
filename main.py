import sys
from PyQt5.QtWidgets import QApplication
from meshit_workflow_gui import MeshItWorkflowGUI

def main():
    app = QApplication(sys.argv)
    window = MeshItWorkflowGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()