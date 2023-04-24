import cv2
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Load dataset
dataset_path = 'C:/Python27/imagen/dataSet'
X = []
y = []
label_map = {}
label_id = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith("jpg"):
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            X.append(img)
            label = root.split('\\')[-1]
            if label not in label_map:
                label_map[label] = label_id
                label_id += 1
            y.append(label_map[label])


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create LBP object
lbp = cv2.face.LBPHFaceRecognizer_create()

# Train LBP model
lbp.train(X_train, np.array(y_train).astype('int'))

# Normalize histograms
X_train = [cv2.equalizeHist(img) for img in X_train]
X_test = [cv2.equalizeHist(img) for img in X_test]


# Flatten each image in X_train and X_test
X_train_flat = [img.reshape(-1) for img in X_train]
X_test_flat = [img.reshape(-1) for img in X_test]

# Create SVM object
svm = SVC(kernel='linear', C=1, probability=True)

# Fit SVM model to LBP features
svm.fit(X_train_flat, y_train)

# Evaluate model on test set
accuracy = svm.score(X_test_flat, y_test)
print('Test accuracy:', accuracy)


lbp.write('lbp_model.yml')
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

np.save('label_map.npy', label_map)



