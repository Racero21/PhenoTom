from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PySide6.QtCore import Signal, Qt, QUrl

class BatchWidget(QWidget):
    batchClicked = Signal(int, str)
    def __init__(self, batch_id, batch_name, delete_callback, parent=None):
        super().__init__(parent)

        self.batch_id = batch_id
        self.batch_name = batch_name

        # Display batch information
        self.batch_label = QLabel(f"Batch: {batch_name}")

        # Delete button
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.deleteButtonClicked)
        
        self.click_button = QPushButton("View Statistics")
        self.click_button.clicked.connect(self.handleButtonClick)
        
        # view images
        self.viewImagesButton = QPushButton("View Images")
        self.viewImagesButton.clicked.connect(self.viewImages)
        
        # Layout for the batch widget
        # self.central_widget = QWidget()
        # self.setCentralWidget(central_widget)
        # layout = QVBoxLayout(self.central_widget)
        layout = QVBoxLayout()
        layout.addWidget(self.batch_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.delete_button)
        layout.addWidget(self.click_button)
        # layout.addWidget(self.viewImagesButton)
        # self.central_widget.setStyleSheet("""
        #     background-color: #ffffff;
        #     border: 10px solid #ccc;
        #     border-radius: 8px;
        #     padding: 10px;
        #                      """)
        # Set layout
        self.setLayout(layout)

        self.delete_callback = delete_callback
    
    def deleteButtonClicked(self):
        if self.delete_callback:
            self.delete_callback(self.batch_id, self.batch_name)
    
    def handleButtonClick(self):
        self.batchClicked.emit(self.batch_id, self.batch_name)

    def viewImages(self):
        output_folder = self.db_handler.getBatchOutputFolderPath(self.batch_id)
        if output_folder:
            QFileDialog.getOpenFileUrl(self, "Open Image Folder", QUrl.fromLocalFile(output_folder), options=QFileDialog.ShowDirsOnly)