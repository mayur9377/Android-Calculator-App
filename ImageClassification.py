from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog as fd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import joblib

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def func():
    image_file_name=fd.askopenfilename()
    clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

    # fit the training data to the model
    clf.fit(trainDataGlobal, trainLabelsGlobal)
    image = cv2.imread(image_file_name)
    cv2.imwrite("E:\\jupyter files\\dataset\\test\\og_image.jpg",image)
    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    rescaled_feature = scaler.transform(global_feature.reshape(-1,1).T)

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]
    global prob
    prob=max(clf.predict_proba(rescaled_feature.reshape(1,-1)).flatten().tolist())

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    #new_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("E:\\jupyter files\\dataset\\test\\new_image.jpg",image)
    # display the output image
    
    global img1,img2
    img2=ImageTk.PhotoImage(Image.open(test_path+"\\new_image.jpg").resize((250,250),Image.ANTIALIAS))
    img1=ImageTk.PhotoImage(Image.open(test_path+"\\og_image.jpg").resize((250,250),Image.ANTIALIAS))
    l1.configure(image=img1)
    l2.configure(image=img2)
    l3.configure(text="Input Image")
    l4.configure(text="Output Image")
    
    l5.configure(text="Accuracy: "+str(prob*100)+"%")
    l6.configure(text="Flower Class: "+train_labels[prediction])
    b1.grid(row=7,column=1)
    b1.configure(text="Select new flower")
    l7.destroy()
    l8.destroy()
    l9.destroy()
    l3.grid(row=2,column=1)
    l4.grid(row=2,column=2)
    l1.grid(row=3,column=1)
    l2.grid(row=3,column=2)
    l5.grid(row=6,column=1)
    l6.grid(row=5,column=1)
    

images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "E:\\jupyter files\\dataset\\train"
h5_data          = 'E:\\jupyter files\\output\\data.h5'
h5_labels        = 'E:\\jupyter files\\output\\labels.h5'
bins             = 8
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
test_size = 0.10
seed      = 9
test_path  = "E:\\jupyter files\\dataset\\test"
scoring    = "accuracy"
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)


#create a gui window using tkinter
gui=Tk()
gui.geometry('510x400')
gui.resizable(False,False)
gui.title("Flower Classification")
b1=Button(gui,text="Select a flower",command=func)
b1.grid(row=4,column=1)
img1=""
img2=""
prob=0.0
l1=Label(gui,image=img1)
l2=Label(gui,image=img2)
l3=Label(gui,text="")
l4=Label(gui,text="")

l5=Label(gui,text="")

l6=Label(gui,text="")

img3=Image.open(r'E:\flower_valley.jpg')
img3=img3.resize(size=(495,300))
ph=ImageTk.PhotoImage(img3)
l7=Label(gui,image=ph)
l7.grid(row=2,column=1)
l8=Label(gui,text="Welcome to Flower Valley")
l8.grid(row=1,column=1)
l9=Label(gui,text="Want a flower? Click below..!")
l9.grid(row=3,column=1)
gui.mainloop()
