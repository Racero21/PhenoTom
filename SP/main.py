from PySide6.QtWidgets import QApplication
from mainwindow import MainWindow
import sys
from qt_material import apply_stylesheet

app = QApplication(sys.argv)
apply_stylesheet(app, theme='dark_teal.xml')
window = MainWindow()
window.show()

app.exec()
