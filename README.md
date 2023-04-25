# Face Recognition using LBP and SVM #

This is a face recognition system that uses OpenCV and scikit-learn. It detects faces in real-time video streams and recognizes them using a combination of Local Binary Patterns (LBP) and Support Vector Machines (SVM).

## Getting Started ##
To use this system, you need to have Python 3 installed, along with the following libraries:

OpenCV
scikit-learn
NumPy
Matplotlib
Pickle
Once you have installed these dependencies, you can download or clone this repository to your local machine.

## Training the System ##
To train the system, you need to create a dataset of faces. The dataset should contain one folder for each person, with each folder containing multiple images of that person's face. The system uses these images to extract LBP features, which are then used to train an SVM classifier.

To create the dataset, you can use any method you like to collect images of faces. Once you have the images, you can use the train.py script to extract LBP features and train the SVM classifier. Here's how to use the script:

## Running the System ##
To run the face recognition system, you can use the recognize.py script. This script uses the trained LBP and SVM models to recognize faces in real-time video streams from your webcam.

Here's how to use the script:
```
python recognize.py
```
When you run the script, your webcam will be initialized, and the system will start recognizing faces in real-time. The recognized faces will be highlighted with a green rectangle, and their names and confidence scores will be displayed next to them.

## Customizing the System ##
The face recognition system can be customized in several ways:

You can change the SVM kernel from linear to Gaussian or radial basis, by modifying the kernel parameter in the SVC constructor in the train.py script.
You can adjust the LBP parameters, such as the radius and number of neighbors, by modifying the radius and neighbors parameters in the cv2.face.LBPHFaceRecognizer_create constructor in the train.py script.
You can adjust the face detection parameters, such as the scale factor and minimum size, by modifying the parameters in the detectMultiScale method of the CascadeClassifier object in the recognize.py script.

## Conclusion ##
This face recognition system provides a simple and customizable way to recognize faces in real-time video streams using OpenCV and scikit-learn. With some customization, it can be adapted to a wide range of face recognition applications.

## Team members ##
--------------------
Gulshan Kumar <br />
_Nitigya Joshi_<br />
Bharath Gujjari<br />
