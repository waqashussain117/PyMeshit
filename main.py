# main.py
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
import os
from PyQt5.QtWidgets import QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

def main():
    try:
        from meshit_workflow_gui import MeshItWorkflowGUI
    except Exception as e:
        print("Failed to import GUI:", e, file=sys.stderr)
        raise

    app = QApplication(sys.argv)
    # set application icon if available
    icon_path = os.path.join(os.path.dirname(__file__), 'resources', 'images', 'app_logo_small.png')
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # show splash screen if image available
    splash_path = os.path.join(os.path.dirname(__file__), 'resources', 'images', 'app_logo.png')
    if os.path.exists(splash_path):
        pixmap = QPixmap(splash_path)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    else:
        # fallback small transparent pixmap
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.transparent)
        splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)

    splash.show()
    app.processEvents()

    # instantiate and show main window while splash is visible
    window = MeshItWorkflowGUI()
    window.show()

    # finish splash and give focus to main window
    splash.finish(window)

    # prefer exec() for modern PyQt, exec_() also works on PyQt5
    sys.exit(app.exec())

if __name__ == "__main__":
    main()