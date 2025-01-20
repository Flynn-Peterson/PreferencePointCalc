import pdfplumber
import re
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

def decode(huntCode):
    """
    Decodes the hunt code into a dictionary containing information about the hunt.

    Args:
    - huntCode (str): The hunt code to be decoded.

    Returns:
    - dict: A dictionary containing information about the hunt, including animal, sex, unit, season, and weapon.
    """
    decoded_info = {}
    # Splitting the hunt code based on the provided format
    decoded_info['Animal'] = 'Deer' if huntCode[0] == 'D' else 'Unknown'
    decoded_info['Sex'] = 'Male' if huntCode[1] == 'M' else 'Female' if huntCode[1] == 'F' else 'Either'
    decoded_info['Unit'] = huntCode[2:5]
    season_mapping = {
        'P1': 'Private First season',
        'P2': 'Private Second season',
        'P3': 'Private Third season',
        'P4': 'Private Fourth season',
        'K1': 'Kids First season',
        'K2': 'Kids Second season',
        'K3': 'Kids Third season',
        'K4': 'Kids Fourth season',
        'W1': 'Ranching First season',
        'W2': 'Ranching Second season',
        'W3': 'Ranching Third season',
        'W4': 'Ranching Fourth season',
        'J1': 'General First season',
        'J2': 'General Second season',
        'J3': 'General Third season',
        'J4': 'General Fourth season',
        'L1': 'Late Season',
        'E1': 'Early Season'
        
    }
    decoded_info['Season'] = season_mapping.get(huntCode[5:7], 'Unknown season')
    weapon_mapping = {
        'R': 'Rifle',
        'A': 'Archery',
        'M': 'Muzzleloader',
        'X': 'Season choice'
    }
    decoded_info['Weapon'] = weapon_mapping.get(huntCode[7], 'Unknown weapon')

    return decoded_info


def parse_data_from_pdf(pdf_path):
    """
    Parses data from a PDF file containing hunt code information.

    Args:
    - pdf_path (str): The path to the PDF file.

    Returns:
    - list: A list containing the parsed data.
    """
    # Regular expressions for pattern matching
    hunt_code_pattern = re.compile(r'\b[A-Z]{2}\d{3}[A-Z]\d[A-Z]\b')
    adult_pattern = r"^(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)$"
    badPattern = r"1\s(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+1\s(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    
    with pdfplumber.open(pdf_path) as pdf:
        data = []
        outData = ''
        
        for page in pdf.pages[3:]:
            outData+=page.extract_text()

        # Replace dashes with zeros
        outData = re.sub("-", "0",outData)
        # Split the text into lines
        lines = outData.strip().splitlines()
        
        # Process each line
        for line in lines:
            #print(line)
            # Check for matches with different patterns
            badmatch = re.search(badPattern,line)
            huntMatch = re.fullmatch(hunt_code_pattern,line)
            goodMatch = re.search(adult_pattern, line)
            # Depending on the matches found, extract relevant data
            if badmatch and not goodMatch and not huntMatch:
                intArr = re.findall(r"\d+",line)
                intArr.pop(0)
                intArr.pop(7)
                data.append(intArr)
            if goodMatch and not badmatch and not huntMatch:
                arr = re.findall(r"\d+",line)
                data.append(arr)
            if huntMatch and not badmatch and not goodMatch:
                data.append(line)
    return data

def getAllHuntCodes(arr):
    """
    Extracts hunt codes from a list of parsed data.

    Args:
    - arr (list): The list of parsed data.

    Returns:
    - list: A list containing all hunt codes.
    """
    codeArr = []
    for item in arr:
        for hunt in item:
            if len(hunt) != 0:
                codeArr.append(hunt[0]['Hunt'])
    return codeArr


def split_by_hunt_code(arr):
    """
    Splits the parsed data into groups based on hunt codes.

    Args:
    - arr (list): The list of parsed data.

    Returns:
    - list: A list containing groups of data, each corresponding to a hunt code.
    """
    code_groups = []
    current_group = []
    for item in arr:
        if isinstance(item, str) and re.match(r'^[A-Z]{2}\d{3}[A-Z]\d[A-Z]$', item):
            if current_group:
                code_groups.append(current_group)
            current_group = [item]
        else:
            current_group.append(item)
    if current_group:  # Append the last group
        code_groups.append(current_group)
    return code_groups

def groupToPPoints(group):
    """
    Formats a group of data into a dictionary.

    Args:
    - group (list): The group of data.

    Returns:
    - dict: A dictionary containing the formatted data.
    """
    formatted_group = []

    hunt_code = group[0]

    for sublist in group[1:]:
        points = sublist[0]
        res_app = sublist[1]
        non_res_app = sublist[2]
        res_tags_given = sublist[8]
        non_res_tags_given = sublist[9]
        hunt_code_str = ''.join(hunt_code)
        formatted_sublist = {
            'Hunt': hunt_code_str,
            'Points': points,
            'Resident Applications': res_app,
            'Non Resident Applications': non_res_app,
            'TagsGiven': {
                'Resident': res_tags_given,
                'Non Resident': non_res_tags_given
            }
        }
        formatted_group.append(formatted_sublist)
    return formatted_group

def printFormatGroup(group):
    """
    Prints the formatted data in a group.

    Args:
    - group (list): The group of formatted data to print.
    """
    for item in group:
        print(item['Hunt'] + ': Hunt with: ' + item['Points'] + ' points and ' + item['Resident Applications'] +  ' res apps then gave:  ' + item['TagsGiven']['Resident']+ ' tags')
        print(item['Hunt'] + ': Hunt with: ' + item['Points'] + ' points and ' + item['Non Resident Applications'] +  ' Non res apps then gave:  ' + item['TagsGiven']['Non Resident']+ ' tags')
        
def calculatePercentGivenResTag(group):
    """
    Calculates the percentage of tags given to residents in a group.

    Args:
    - group (list): The group of formatted data.

    Returns:
    - list: A list containing dictionaries with hunt code, points, and percentage of tags given to residents.
    """
    resultList = []
    
    for item in group:
        arr = {'Hunt': None,'Points': None, 'Percent':None }
        if int(item['Resident Applications']) != 0 and int(item['TagsGiven']['Resident']) != 0:
            percentGiven = (int(item['TagsGiven']['Resident']) / int(item['Resident Applications']))*100
            arr.update({'Hunt': item['Hunt'],'Points': item['Points'], 'Percent' : percentGiven})
            resultList.append(arr)
        if int(item['Resident Applications']) != 0 and int(item['TagsGiven']['Resident']) == 0:
            percentGiven = 0
            arr.update({'Hunt': item['Hunt'],'Points': item['Points'], 'Percent' : percentGiven})
            resultList.append(arr)
    return resultList
    
        
def calculatePercentNonRes(group):
    """
    Calculates the percentage of tags given to non-residents in a group.

    Args:
    - group (list): The group of formatted data.

    Returns:
    - list: A list containing dictionaries with hunt code, points, and percentage of tags given to non-residents.
    """
    resultList = []
    
    for item in group:
        arr = {'Hunt': None,'Points': None, 'Percent':None }
        if int(item['Non Resident Applications']) != 0 and int(item['TagsGiven']['Non Resident']) != 0:
            percentGiven = (int(item['TagsGiven']['Non Resident']) / int(item['Non Resident Applications']))*100
            arr.update({'Hunt': item['Hunt'],'Points': item['Points'], 'Percent' : percentGiven})
            resultList.append(arr)
        if int(item['Non Resident Applications']) != 0 and int(item['TagsGiven']['Non Resident']) == 0:
            percentGiven = 0
            arr.update({'Hunt': item['Hunt'],'Points': item['Points'], 'Percent' : percentGiven})
            resultList.append(arr)
    return resultList

def showGraph(data):
    all_points = []
    all_percentages = []

    data.reverse()
    for i in range(len(data)):
        all_percentages.append(2015 + i)
        all_points.append(data[i])
    # Plot all data points at once
    plt.plot(all_percentages ,all_points, label='Percentage of Tags Given')

    # Set labels and title
    plt.xlabel('Year')
    plt.ylabel('Points')
    plt.title('Points To Guarntee tag')
    plt.legend()

    # Show the combined plot
    plt.show()

def getAllHunts(array,huntCode):
    listOfHunts = []
    for entries in array:
        yearHunts = []  # Inner list for hunts of each year
        for entry in entries:
            for hunt in entry:
                if hunt['Hunt'] == huntCode:
                    yearHunts.append(hunt) 
        listOfHunts.append(yearHunts)  # Append the list of hunts for the current year      
    return listOfHunts

def graphPercents(deList):
    xPlot = []
    yPlot = []
    for i in range(len(deList)):
        yPlot.append(deList[i]['Percent'])
        xPlot.append(deList[i]['Year'])
        
    plt.plot(xPlot,yPlot, label = 'HekYea')
    plt.ylim(-5,105)
    plt.show()
    
    
def calculatePointsNeed(years):
    pointsPerYear = []
    for year in years:
        pointMax = 0
        for i in range(len(year) - 1, -1, -1):
            if int(year[i]['Percent']) < 100.0:
                pointMax = max(pointMax, int(year[i]['Points'])) + 1
        pointsPerYear.append(pointMax)
    return(pointsPerYear)    


def calcPercentBasedOnPoints(points, years):
    percentPerYear = []
    whatYearItIs=2023
    for year in years:
        for i in range(len(year)):
            if int(year[i]['Points']) == points:
                percentPerYear.append({'Year' : whatYearItIs,'Percent' : year[i]['Percent']})
        whatYearItIs-=1
    return percentPerYear
            
            
def prepare_data(hunt_code, points, all_data):
    """
    Prepares the data for linear regression based on a specific hunt code and input number of points.

    Args:
    - hunt_code (str): The hunt code for which to prepare the data.
    - points (int): The number of points for which to predict the percentage.
    - all_data (list): All the data collected from different years.

    Returns:
    - tuple: A tuple containing the feature matrix (X) and target vector (y).
    """
    X = []  # Feature matrix (year)
    y = []  # Target vector (percentage)

    hunt_data = getAllHunts(all_data, hunt_code)
    
    for year_data in hunt_data:
        # Assume year is the index of the year
        for year, data_point in enumerate(year_data):
            # Extract year and percentage
            year = 2015 + year  # Assuming the year starts from 2015
            percent = data_point['Percent']
            
            # Only consider non-zero percentages
            if percent > 0:
                X.append([year])
                y.append(percent)

    X = np.array(X)  # Feature matrix (year)
    y = np.array(y)  # Target vector (percentage)

    return X, y

def polynomial_regression(X, y, degree):
    """
    Fits a polynomial regression model to the data.

    Args:
    - X (ndarray): Feature matrix (points).
    - y (ndarray): Target vector (percentages).
    - degree (int): Degree of the polynomial.

    Returns:
    - LinearRegression: Fitted polynomial regression model.
    """
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    return model

def predict_percentage(poly_reg_model, points_to_predict):
    """
    Predicts the percentage for the provided number of points using the polynomial regression model.

    Args:
    - poly_reg_model (LinearRegression): Fitted polynomial regression model.
    - points_to_predict (int): Number of points to predict the percentage.

    Returns:
    - float: Predicted percentage.
    """
    year_to_predict = np.array([[points_to_predict]])
    predicted_percentage = poly_reg_model.predict(year_to_predict)
    return predicted_percentage[0]



def findHunt(huntCode,points,allData):
    huntMap = {
        'HuntCode' : huntCode,
        'Points' : points,
        'PercentList' : []
    }
    percentList = []
    for data in allData:
        for year in data:
            if len(year) > 0:
                if year[0]['Hunt'] == str(huntCode):
                    for i in range(len(year)):
                        if int(year[i]['Points']) == points:
                            percentList.append(year[i]['Percent'])
    print(percentList)
    huntMap['PercentList'] =percentList
    return huntMap    
    


def predict_future(hunt_code, points_to_predict, all_data, future_years):
    X_train, y_train = prepare_data(hunt_code, points_to_predict, all_data)
    
    # Prepare future data
    future_data = np.array([[year] for year in future_years])  # Assuming year is the feature
    
    # Train your model
    model = train_model(X_train, y_train)  # You'll need to define this function
    
    # Make predictions for the future
    predicted_percentages = model.predict(future_data)
    
    # Visualize results
    plt.plot(future_years, predicted_percentages, label='Predicted Percentages')
    plt.xlabel('Year')
    plt.ylabel('Predicted Percentage')
    plt.title('Predicted Percentages for Hunt Code {}'.format(hunt_code))
    plt.legend()
    plt.show()
        
        
def getData():
    """
    Main function to process the PDF files and perform data analysis.
    """
    # Paths to the PDF files
    pdf_paths = [
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2015DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2016DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2017DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2018DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2019DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2020DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2021DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2022DeerDrawRecap.pdf',
        r'C:\\PersonalProjects\\PreferenceCalc\\DeerData\\2023DeerDrawRecap.pdf'
    ]
    

    allResData = []
    allNonResData= []
    # Process each PDF file
    for pdf_path in pdf_paths:
        print(pdf_path)
        data = parse_data_from_pdf(pdf_path)
        groups = split_by_hunt_code(data)
        usableList = []
        for group in groups[1:]:
            formatGroup = groupToPPoints(group)
            percentList = calculatePercentNonRes(formatGroup)
            usableList.append(percentList)
        allNonResData.append(usableList)
        usableList2 = []
        for group in groups[1:]:
            formatGroup = groupToPPoints(group)
            percentList = calculatePercentGivenResTag(formatGroup)
            usableList2.append(percentList)
        allResData.append(usableList2)
        
    return allResData,allNonResData

def save_data_to_file(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # Write each item to a new line in the file

def main():
    resData,nonResData = getData()
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023])
    
    resPercents = []
    huntsRes = getAllHuntCodes(resData)
    for hunts in huntsRes:
        for i in range(30):
            percentages = findHunt(hunts,i,resData)
            
            if len(percentages['PercentList']) > 0:
                resPercents.append(percentages)
    
    save_data_to_file(resPercents, 'resData.txt')
    
    huntsNon = getAllHuntCodes(nonResData)
    nonresPercents = []
    
    for hunts in huntsNon:
        for i in range(50):
            percentages = findHunt(hunts,i,nonResData)
            if len(percentages['PercentList']) > 0:
                nonresPercents.append(percentages)
    save_data_to_file(nonresPercents, 'nonresData.txt')
    

if __name__ == "__main__":
    main()
