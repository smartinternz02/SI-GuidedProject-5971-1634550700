#import the necessary packages 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import cv2
import os
import pickle #import the pickle file`

def quantify_image(image):
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0,255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features=quantify_image(image)
        data.append(features)
        labels.append(label)
    return (np.array(data), np.array(labels))

trainingPath = r"D:\lekshmi\parkinson disease\Detecting_Parkinson_Disease_Using_ML-main\Flask_App\spiral\training"
testingPath = r"D:\lekshmi\parkinson disease\Detecting_Parkinson_Disease_Using_ML-main\Flask_App\spiral\testing"
trainingPath = r"D:\lekshmi\parkinson disease\Detecting_Parkinson_Disease_Using_ML-main\Flask_App\wave\training"
testingPath = r"D:\lekshmi\parkinson disease\Detecting_Parkinson_Disease_Using_ML-main\Flask_App\wave\testing"
print("[INFO]) loading data...")
(X_train, y_train) = load_split(trainingPath)
(X_test, y_test) = load_split(testingPath)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
print(X_train.shape,y_train.shape)

print("[INFO]) training model...")
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

testingPaths = list(paths.list_images(testingPath))
idxs = np.arange(0, len(testingPaths))
idxs = np.random.choice(idxs, size=(25,), replace=False)
image = []

predictions = model.predict(X_test)
cm = confusion_matrix(y_test, predictions).flatten()
print(cm)
(tn, fp, fn, tp) = cm
accuracy = (tp + tn) / float(cm.sum())
print(accuracy)
pickle.dump(model,open('parkinson.pkl','wb'))