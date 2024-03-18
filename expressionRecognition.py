# Import Necessary Libraries
import joblib
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import math

# Function to print Evaluations
def PrintEvalMetrics(pred, indices, y):
    #manually merge predictions and testing labels from each of the folds to make confusion matrix
    finalPredictions = []
    groundTruth = []
    for p in pred:
        finalPredictions.extend(p)
    for i in indices:
        groundTruth.extend(y[i])
    # Print the precision, recall, and accuracy scores 
    print(confusion_matrix(finalPredictions, groundTruth))
    print("Precision: ", precision_score(groundTruth, finalPredictions, average='macro'))
    print("Recall: ", recall_score(groundTruth, finalPredictions, average='macro'))
    print("Accuracy: " , accuracy_score(groundTruth, finalPredictions))


# Function to retrieve data from a directory
def GetData(directoryName):
    # Set the root directory
    root_folder = directoryName
    # Create empty lists to store data and file paths
    data_list = []
    file_paths = []
    # Set numpy options for printing arrays(This is for printing the data in the command line)
    np.set_printoptions(precision=8, suppress=True,threshold=np.inf)
    # Loop through all files in the directory and its subdirectories
    for dirpath, dirnames, filenames in sorted(os.walk(root_folder)):
        for filename in sorted(filenames):
            # Only load files with .bnd extension
            if os.path.splitext(filename)[1] == '.bnd':
                file_path = os.path.join(dirpath, filename)
                # Load the data from the file
                with open(file_path, 'r') as file:
                    data = np.loadtxt(file, usecols=(1,2,3))
                    # Add the data and file path to the lists
                    data_list.append(data)
                    file_paths.append(file_path)
    # Convert the data list to a numpy array
    data_array = np.array(data_list)
    # Flatten the data array and convert it to a 2D numpy array
    flat_data_list = []
    for element in data_array:
        flat_data_list.append(element.flatten())
    flat_data_array = np.array(flat_data_list)

    # Create a dictionary to map emotions to numbers
    emotions = {"Angry": 0, "Disgust": 1, "Fear": 2, "Happy": 3, "Sad": 4, "Surprise": 5}
    file_array = []
    # Loop through all subdirectories in the root directory
    for folder in sorted(os.listdir(root_folder)):
        if os.path.isdir(os.path.join(root_folder, folder)):
            # Loop through all subdirectories in the current subdirectory
            for subfolder in sorted(os.listdir(os.path.join(root_folder, folder))):
                if os.path.isdir(os.path.join(root_folder, folder, subfolder)):
                    # Loop through all files in the current subdirectory
                    for file in sorted(os.listdir(os.path.join(root_folder, folder, subfolder))):
                        if file.endswith(".bnd"):
                            #Get the .bnd file folder name and store it
                            emotion = subfolder.capitalize()
                            file_array.append(emotions[emotion])
    # Convert the list of target values to a numpy array
    target_array = np.array(file_array)

    return flat_data_array,target_array

# Define a function to perform cross-validation
def CrossFoldValidation( array1_to_pass, array2_to_pass,classifier="SVM",type="Original"):

    X = array1_to_pass
    y = array2_to_pass

    # Initialize a classifier depending on the input parameter
    clf = None
    if classifier == "SVM":
        clf = svm.SVC()
        # print("SVM")
    elif classifier == "RF":
        clf = RandomForestClassifier()
        # print("RF")
    elif classifier == "TREE":
        clf = tree.DecisionTreeClassifier()
        # print("TREE")
    pred=[]
    test_indices=[]
    #10-fold cross validation
    kf = KFold(n_splits=10)
    # Loop through each fold
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        #train classifier
        clf.fit(X[train_index], y[train_index])
        #get predictions and save
        pred.append(clf.predict(X[test_index]))
        #save current test index
        test_indices.append(test_index)
    # Save trained model to a file
    file_name = "file_{}_{}.pkl".format(classifier,type)
    joblib.dump(clf, file_name)
    return pred, test_indices, y

# Modify the data based on given input
def DataType(array1_to_pass,array2_to_pass,type="Original"):
    # If type is Original, simply return the arrays as they are.
    if type=="Original":
        # print("**************************************Original**************************************")
        # print(array1_to_pass[0])
        return array1_to_pass,array2_to_pass
    # If type is Translated, perform mean centering of the data and return the processed arrays.
    elif type=="Translated":
        # print("**************************************Translated**************************************")
        # Create a copy of the input array and reshape it to have 83 columns.
        my_array = array1_to_pass.copy()
        num_rows, num_cols = my_array.shape
        reshaped_array = np.zeros((num_rows, 83, 3))
        for i in range(num_rows):
            reshaped_array[i] = my_array[i].reshape(83, 3)
        # Calculate the mean of each column for each row and subtract from each element
        for i in range(num_rows):
            col_means = np.mean(reshaped_array[i], axis=0)
            reshaped_array[i] -= col_means
        # Reshape back to original form
        my_array = reshaped_array.reshape(num_rows, -1)
        return my_array,array2_to_pass
    # If type is RotatedX, rotate the input array by 180 degrees along the X-axis and return the processed arrays.
    elif type=="RotatedX":
        # print("**************************************RotatedX**************************************")
        Pi=round(2*math.acos(0.0), 3)
        array1_to_pass = np.round(array1_to_pass, 6)
        for i in range(array1_to_pass.shape[0]):
            row = array1_to_pass[i]
            for j in range(0, len(row)-2, 3):
                row[j+1] = math.cos(Pi) * row[j+1] + math.sin(Pi) *  row[j+2] 
                row[j+2] = -math.sin(Pi) * row[j+1] + math.cos(Pi) * row[j+2]
        return array1_to_pass,array2_to_pass   
    # If type is RotatedY, rotate the input array by 180 degrees along the Y-axis and return the processed arrays.     
    elif type=="RotatedY":
        # print("**************************************RotatedY**************************************")
        Pi=round(2*math.acos(0.0), 3)
        array1_to_pass = np.round(array1_to_pass, 6)
        for i in range(array1_to_pass.shape[0]):
            row = array1_to_pass[i]
            for j in range(0, len(row)-2, 3):
                row[j] = math.cos(Pi) * row[j] - math.sin(Pi) *  row[j+2] 
                row[j+2] = math.sin(Pi) * row[j] + math.cos(Pi) * row[j+2]
        return array1_to_pass,array2_to_pass
    # If type is RotatedZ, rotate the input array by 180 degrees along the Z-axis and return the processed arrays.
    elif type=="RotatedZ":
        # print("**************************************RotatedZ**************************************") 
        Pi=round(2*math.acos(0.0), 3)
        array1_to_pass = np.round(array1_to_pass, 6)
        for i in range(array1_to_pass.shape[0]):
            row = array1_to_pass[i]
            for j in range(0, len(row)-2, 3):
                row[j] = math.cos(Pi) * row[j] + math.sin(Pi) *  row[j+1] 
                row[j+1] = -math.sin(Pi) * row[j] + math.cos(Pi) * row[j+1]
        return array1_to_pass,array2_to_pass
    return array1_to_pass,array2_to_pass  
    
# This is for plotting the Face Data
def Plot(data_array):
    axis = data_array[0].reshape(-1, 3)
    # print(axis)
    # Extract x, y, and z columns
    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # Set axis labels and title
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('3D Scatter Plot')
    # Show the plot
    plt.show()

# Create argument parser object
parser = argparse.ArgumentParser(description='Demo for Iris dataset classification')
# Add command line arguments with their data types, default values and help descriptions
parser.add_argument('classifier', nargs='?', type=str, default='SVM', help='Classifier type; if none given, SVM is default.')
parser.add_argument('type', nargs='?', type=str, default='Original', help='Data Type: If not given Original is default')
parser.add_argument('directoryName', type=str, default='BU4DFE_BND_V1.1', help='Directory Name: If not given BU4DFE_BND_V1.1 is default')
# Parse the command line arguments and store them in an object called "args"
args = parser.parse_args()
# Call to GetData() function to get data from the directory specified in the command line arguments
array1_to_pass,array2_to_pass = GetData(args.directoryName)
# Call to DataType() function to transform the data based on the data type specified in the command line arguments
type_array1_to_pass,type_array2_to_pass = DataType(array1_to_pass,array2_to_pass,args.type)
# Call to CrossFoldValidation() function to perform cross-fold validation and get predictions, test indices, and actual labels
pred, test_indices, y = CrossFoldValidation(type_array1_to_pass,type_array2_to_pass,args.classifier,args.type)
# Call to PrintEvalMetrics function to print evaluation metrics based on the predictions, test indices, and actual labels
PrintEvalMetrics(pred, test_indices, y)