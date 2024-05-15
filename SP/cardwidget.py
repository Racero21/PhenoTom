import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget

class CardExample(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a vertical layout for the card
        layout = QVBoxLayout(central_widget)

        # Add a label (you can customize this with images, buttons, etc.)
        label = QLabel("This is a card-like widget")
        layout.addWidget(label)

        # Set some styles to make it look like a card
        central_widget.setStyleSheet("""
            background-color: #ffffff;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 10px;
        """)

        self.setWindowTitle("Card Example")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CardExample()
    window.show()
    sys.exit(app.exec())