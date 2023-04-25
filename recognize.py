import cv2
import numpy as np
import os
import pickle


# Load trained LBP model
lbp = cv2.face.LBPHFaceRecognizer_create()
lbp.read('lbp_model.yml')

# Load trained SVM model
with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Load label map
label_map = np.load('label_map.npy', allow_pickle=True).item()

# Initialize camera capture
video_capture = cv2.VideoCapture(0)

# Loop over frames from the camera
while True:
    # Capture a frame from camera
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1, 0)  # Flip frame to show mirror image

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Extract face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Resize ROI to match training image size
        roi_resized = cv2.resize(roi_gray, (100, 100), interpolation=cv2.INTER_LINEAR)

        # Predict label for ROI using LBP and SVM models
        label_id, confidence = lbp.predict(roi_resized)
        # print(label_id, confidence)
        label = list(label_map.keys())[list(label_map.values()).index(label_id)]
        #if label is not in the label_map set label as unknown
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        confidence = confidence/100 -0.8
        cv2.putText(frame, '{} {:.2f}'.format(label, confidence), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the resulting frame
    cv2.imshow('Video', frame)

    # Quit program when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
video_capture.release()
cv2.destroyAllWindows()
