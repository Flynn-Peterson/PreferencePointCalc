import re
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt6 import uic
import sys
import numpy as np
import matplotlib.pyplot as plt
import HuntCodeDecoder
from sklearn.svm import SVR


class UI(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load the .ui file
        ui_file = "HuntAPP2.ui"
        Ui_MainWindow, QtBaseClass = uic.loadUiType(ui_file)

        # Create an instance of the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect button click signal to function
        self.ui.pushButton_2.clicked.connect(self.on_button_clicked)
        
    def validateInputs(self):
        year = self.ui.Years.toPlainText()
        point = self.ui.Points.toPlainText()
        hunt = self.ui.Hunt.toPlainText()
        res = self.ui.Res.toPlainText()

        if not year.strip() or not point.strip() or not hunt.strip() or not res.strip():
            print("Please fill in all fields.")
            # Clear fields if necessary
            self.ui.Years.clear()
            self.ui.Points.clear()
            self.ui.Hunt.clear()
            self.ui.Res.clear()
            return None
        else:
            # All fields are filled, return the data
            return {'year': year, 'point': point, 'hunt': hunt, 'res': res}
        
    def display(self, year , point, hunt, res):
        degree = 2
        years = [2015, 2016, 2017, 2018, 2019, 2020,2021,2022,2023]
        percentages = []
        if res == 'non':
            percentages = search_hunt_code('nonresData.txt',hunt,point)
        elif res == 'res':
            percentages = search_hunt_code('resData.txt',hunt,point)
        else:
            QMessageBox.critical(self, "Failed", "Invalid residency.", QMessageBox.StandardButton.Ok)
        
        if len(percentages) == 0:
            QMessageBox.critical(self, "Failed", "Invalid Hunt.", QMessageBox.StandardButton.Ok)
        else:
            while len(percentages) < len(years):
                prev = percentages[-1]
                percentages.append(prev)

        percent = percentages[::-1]
        
        print(percent)
        coefficients = np.polyfit(years, percent, degree)
        year = int(year)
        # Make predictions for future years
        future_years = np.arange(2024, year)  # example: predict for the next 6 years
        predicted_percentages = np.polyval(coefficients, future_years)

        min_threshold = 0  # Example minimum threshold value
        max_threshold = 100  # Example maximum threshold value
        adjusted_predicted_percentages = []

        for percentage in predicted_percentages:
            adjusted_percentage = max(min(percentage, max_threshold), min_threshold)  # Apply both thresholds
            adjusted_predicted_percentages.append(adjusted_percentage)

        # Plot the original data and the adjusted predicted values
        plt.plot(years, percentages, 'bo-', label='Actual Data')
        plt.plot(future_years, adjusted_predicted_percentages, 'ro--', label='Adjusted Predicted Data')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.title('Adjusted Percentage Prediction for Future Years')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Display the adjusted predicted percentages for future years
        for year, percentage in zip(future_years, adjusted_predicted_percentages):
            print(f'Predicted percentage for year {year}: {percentage:.2f}%')
    def on_button_clicked(self):
        print("Button clicked!")  # Placeholder action, replace with desired functionality
        data = self.validateInputs()
        if data is not None:
            self.display(data['year'], data['point'], data['hunt'], data['res'])
        else:
            print("Inputs are not valid. Please fill in all fields.")
            # Display a failed message dialog
            QMessageBox.critical(self, "Failed", "Please fill in all fields.", QMessageBox.StandardButton.Ok)
        
        
        

def search_hunt_code(file_path, hunt_code, points):
    percentages = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = eval(line)
                    if data['HuntCode'] == str(hunt_code):
                        print(hunt_code)
                        print(data['Points'])
                        print(points)
                        
                
                        if int(data['Points']) == int(points):
                            print(points)
                            percentages = data['PercentList']
                            print(percentages)
                            return percentages
                except SyntaxError:
                    print("error")
            else:
                print("Hunt code not found in the file.")
    except FileNotFoundError:
        print("File not found.")
    return percentages

if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = UI()
    main_window.show()

    sys.exit(app.exec())