from PySide6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea, QInputDialog, QMessageBox, QGridLayout, QSizePolicy, QLineEdit
from PySide6.QtCore import Qt
from statisticsview import StatisticsView
from databasehandler import DatabaseHandler
from batchwidget import BatchWidget
from final import extract_parameters
from test import getCoinScale
import numpy as np
import os

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("TomatoVision")

        # Create a button
        self.button = QPushButton("Add images")
        self.button.clicked.connect(self.openFileDialog)

        self.batch_layout = QGridLayout()

        # Create a central widget
        central_widget = QWidget()
        # central_widget.setLayout(layout)

        scroll_area = self.createScrollArea()

        # Create a layout
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.createSearchBar())
        layout.addWidget(self.button)
        layout.addLayout(self.batch_layout)
        layout.addWidget(scroll_area)
        # grid_layout = QGridLayout(central_widget)
        # grid_layout.setAlignment(Qt.AlignTop)
        # grid_layout.addWidget(self.button)
        # grid_layout.addLayout(self.batch_layout)
        # grid_layout.addWidget(scroll_area)

        # Set the central widget
        self.setCentralWidget(central_widget)

        self.statistics_widget = QWidget()
        self.statistics_layout = QVBoxLayout(self.statistics_widget)
        self.statistics_widget.hide()
        
        self.db_handler = DatabaseHandler()

        self.displayExistingBatches()

        self.resize(800, 600)

        self.resizeEvent = self.handleResize

    def handleResize(self, event):
        self.updateBatchWidgetSizes()

    def createSearchBar(self):
        search_bar = QLineEdit()
        search_bar.setPlaceholderText("Search...")
        search_bar.textChanged.connect(self.filterBatchWidgets)
        return search_bar

    def filterBatchWidgets(self, text):
        text = text.lower()
        print(self.batch_layout.count())
        for i in range(self.batch_layout.count()):
            widget = self.batch_layout.itemAt(i).widget()
            if text in widget.batch_name.lower():
                widget.show()
            else:
                widget.hide()

    def openFileDialog(self):
    # ask for name
        get_name, ok_pressed = QInputDialog.getText(self, 'Add new batch', 'Enter name: ')
        if ok_pressed: 
            print(get_name)
            
            # Open file dialog to select image files
            file_dialog = QFileDialog()
            file_dialog.setWindowTitle("Select Image Files")
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.gif *.tif)")

            batch_id = self.db_handler.insertBatch(get_name)

            if file_dialog.exec_() == QFileDialog.Accepted:
                # Process selected image files
                selected_files = file_dialog.selectedFiles()

                # Display a modal to choose if a scale will be used
                scale_used = QMessageBox.question(self, 'Use Scale', 
                                                'Would you like to use a scale for these images?',
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                coin_diameter = None
                use_one_diameter = False
                if scale_used == QMessageBox.Yes:
                    # Ask if the same diameter should be used for all images
                    use_one_diameter = QMessageBox.question(self, 'Same Diameter for All Images', 
                                                            'Do you want to use the same coin diameter for all images?',
                                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

                    if use_one_diameter == QMessageBox.Yes:
                        # Ask for the diameter of the coin once
                        coin_diameter, diameter_ok = QInputDialog.getDouble(self, 'Coin Diameter', 
                                                                            'Enter the diameter of the coin (in mm):', 
                                                                            decimals=10)
                        if not diameter_ok:
                            coin_diameter = None

                for file_name in selected_files:
                    print(f"Selected file: {file_name}")
                    
                    parameters = self.extractAndStoreParameters(batch_id, get_name, file_name, coin_diameter)

                    # # If using one diameter for all images
                    # if use_one_diameter == QMessageBox.Yes and coin_diameter is not None:
                    #     getCoinScale(file_name, coin_diameter)
                    # elif scale_used == QMessageBox.Yes and use_one_diameter == QMessageBox.No:
                    #     # Ask for the diameter of the coin for each image
                    #     coin_diameter, diameter_ok = QInputDialog.getDouble(self, 'Coin Diameter', 
                    #                                                         f'Enter the diameter of the coin for {file_name} (in mm):', 
                    #                                                         decimals=2)
                    #     if diameter_ok:
                    #         getCoinScale(file_name, coin_diameter)

                    # self.db_handler.insertImagePath(batch_id, file_name)

            # self.displayStatisticsView(batch_id)
            self.displayExistingBatches()

    def extractAndStoreParameters(self, batch_id, batch_name, image_path, coin_diameter):
        # set output folder for each batch
        output_folder = os.path.join("output", f"batch_{batch_name}_{batch_id}")
        os.makedirs(output_folder, exist_ok=True)

        adjusted = []

        parameters = extract_parameters(image_path, output_folder)
        coin = getCoinScale(image_path)
        print(f'coin is {coin}')


        area_conversion_factor = (coin_diameter / coin) ** 2
        for i, element in enumerate(parameters):
            if i == 0:  # First element is the area
                element = element * area_conversion_factor
            elif i == 3:
                element = element
            elif i == len(parameters) - 1:  # Last element is the convex hull area
                element = element * area_conversion_factor
            elif coin_diameter is not None:
                element = element * (coin_diameter / coin)  # Adjust other elements if needed
            element = round(element, 2)
            adjusted.append(float(element))
            print(f'{element} YOEWWWW')
        # self.db_handler.insertImagePath(batch_id, image_path, parameters[0], parameters[1], parameters[2], parameters[3], parameters[4])
        self.db_handler.insertImagePath(batch_id, image_path, adjusted[0], adjusted[1], adjusted[2], adjusted[3], adjusted[4])
        print(f" \n FROM EXTRACTION ----------->>>>  {adjusted[0]} {adjusted[1]} {adjusted[2]} {adjusted[3]} {adjusted[4]} ")
        return parameters

    def displayStatisticsView(self, batch_id, batch_name):
        for i in reversed(range(self.statistics_layout.count())):
            self.statistics_layout.itemAt(i).widget().setParent(None)

        # Extract parameters for the batch and add them to the statistics widget
        parameters = self.db_handler.getBatchParameters(batch_id)

        if parameters:
            statistics_view = StatisticsView(batch_id, batch_name, parameters)
            self.statistics_layout.addWidget(statistics_view)
        # Show the statistics widget
        self.statistics_widget.show()

    def displayExistingBatches(self):
        for i in reversed(range(self.batch_layout.count())):
            self.batch_layout.itemAt(i).widget().setParent(None)

        batches = self.db_handler.getExistingBatches()
        
        num_batches = len(batches)
        num_columns = 2
        num_rows = (num_batches + num_columns - 1) // num_columns

        for i, (batch_id, batch_name) in enumerate(batches):
            # Calculate the row and column for the current batch
            row = i // num_columns
            col = i % num_columns

            batch_widget = BatchWidget(batch_id, batch_name, self.deleteBatch)
            batch_widget.batchClicked.connect(self.displayStatisticsView)
            
            # Add the batch widget to the grid layout
            self.batch_layout.addWidget(batch_widget, row, col)

            # Set the size policy for the batch widget (optional)
            size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            batch_widget.setSizePolicy(size_policy)

        # for batch_id, batch_name in batches:
        #     batch_widget = BatchWidget(batch_id, batch_name, self.deleteBatch)
        #     batch_widget.batchClicked.connect(self.displayStatisticsView)
        #     self.batch_layout.addWidget(batch_widget)

        #     batch_widget.setFixedSize(200,100)
        #     self.batch_layout.setSpacing(10)

            # batch_widget.setFixedSize(desired_width, desired_height)

            # self.batch_layout.addChildWidget(batch_widget)
        
        self.updateBatchWidgetSizes()

    def updateBatchWidgetSizes(self):
        window_size = self.size()
        desired_width = max(int(window_size.width() * 0.05), 200)
        desired_height = max(int(window_size.height() * 0.05), 100)

        for i in range(self.batch_layout.count()):
            widget = self.batch_layout.itemAt(i).widget()
            widget.setFixedSize(desired_width, desired_height)

    def deleteBatch(self, batch_id, batch_name):
        reply = QMessageBox.question(
            self,
            "Delete Batch",
            f"Do you want to delete Batch {batch_name} with id {batch_id}? This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.db_handler.deleteBatch(batch_id, batch_name)

            self.displayExistingBatches()

    def createScrollArea(self):
        scroll_area = QWidget()
        scroll_layout = QVBoxLayout(scroll_area)
        scroll_layout.addLayout(self.batch_layout)

        scroll_area_widget = QWidget()
        scroll_area_widget.setLayout(scroll_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidget(scroll_area_widget)
        scroll_area.setWidgetResizable(True)
        # scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        return scroll_area