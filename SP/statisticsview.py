from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QFileDialog, QApplication
from PySide6.QtGui import QClipboard
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import csv

class StatisticsView(QWidget):
    def __init__(self, batch_id, batch_name, parameters, parent=None):
        super().__init__(parent)

        self.batch_id = batch_id
        self.batch_name = batch_name
        self.parameters = parameters

        # Add a label for batch information
        self.statistics_label = QLabel(f"Data for Batch {batch_name}")

        # Dropdown menu for parameter selection
        self.parameter_selector = QComboBox()
        self.parameter_selector.addItems(['Projected Area', 'Extent X', 'Extent Y', 'Eccentricity', 'Convex Hull Area'])
        self.parameter_selector.currentIndexChanged.connect(self.updatePlot)

        # Layout for the statistics view
        layout = QVBoxLayout()
        layout.addWidget(self.statistics_label)

        # export button
        self.export_button = QPushButton('Export')
        self.export_button.clicked.connect(self.exportToCSV) 

        # Add a button to copy the selected column to the clipboard
        self.copy_column_button = QPushButton('Copy Column')
        self.copy_column_button.clicked.connect(self.copyColumnToClipboard)
        
        # Create a Matplotlib figure and canvas
        self.figure = Figure(figsize=(5, 3), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        layout.addWidget(self.parameter_selector)
        layout.addWidget(self.export_button)
        layout.addWidget(self.copy_column_button)

        # Create a subplot for the line plot
        self.ax = self.figure.add_subplot(111)

        # Initial plot
        self.updatePlot()

        # Set layout
        self.setLayout(layout)

    def updatePlot(self):
        # Extract selected parameter for plotting
        selected_parameter = self.parameter_selector.currentText()
        parameter_index = {'Projected Area': 3, 'Extent X': 4, 'Extent Y': 5, 'Eccentricity': 6, 'Convex Hull Area': 7}[selected_parameter]
        parameter_values = [parameter[parameter_index] for parameter in self.parameters]

        # Clear existing plot
        self.ax.clear()

        # Create a new line plot
        self.ax.scatter(range(1, len(parameter_values) + 1), parameter_values, marker='o')
        self.ax.set_xlabel('Image Sample')
        self.ax.set_ylabel(selected_parameter)

        # Redraw canvas
        self.canvas.draw()
    
    def exportToCSV(self):
        # Open a file dialog to get the desired file path for saving
        file_dialog = QFileDialog()
        file_dialog.setWindowTitle("Save CSV File")
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV Files (*.csv)")

        if file_dialog.exec_() == QFileDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]

            # Write data to CSV file
            with open(file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)

                # Write header
                header = ['Image Index', 'Projected Area', 'Extent X', 'Extent Y', 'Eccentricity', 'Convex Hull Area']
                csv_writer.writerow(header)

                # Write data rows
                for i, parameter in enumerate(self.parameters):
                    row = [i + 1] + [parameter[j] for j in range(3, 8)]  # Indices 3 to 7 are the parameters
                    csv_writer.writerow(row)

            print(f"Data exported to {file_path}")

    def copyColumnToClipboard(self):
        # Get the selected column index
        selected_column_index = self.parameter_selector.currentIndex() + 3  # Offset by 3 to match parameter indices

        # Extract the selected column data
        column_data = [parameter[selected_column_index] for parameter in self.parameters]

        # Convert data to a string
        column_string = '\n'.join(map(str, column_data))

        # Set the string to the clipboard
        clipboard = QApplication.clipboard()
        clipboard.setText(column_string, mode=QClipboard.Clipboard)

        print(f"Column copied to clipboard:\n{column_string}")