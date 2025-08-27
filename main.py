# main.py
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
import os
from PySide6.QtWidgets import QSplashScreen
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

def main():
    try:
        from Pymeshit_workflow_gui import MeshItWorkflowGUI
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

    # prefer exec() for modern PyQt, exec_() also works on PySide6
    sys.exit(app.exec())

if __name__ == "__main__":
    main()