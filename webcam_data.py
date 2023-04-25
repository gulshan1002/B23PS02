import cv2
import time
import os


video_capture = cv2.VideoCapture(0)  # Initialize camera capture

name = input('Enter your name: ')  # Ask user for name to create directory

# Create directory to save the images
new_path = 'C:/Python27/imagen/dataSet'
new_path = new_path + '/' + name

# Check if directory exists, create it if not
if not os.path.exists(new_path):
    os.makedirs(new_path)

# Capture a frame from camera and show it to user to get ready
ret, frame = video_capture.read()
# Flip frame to show mirror image
frame = cv2.flip(frame, 1, 0)
cv2.imshow('Frame', frame)
print('Press any key to begin')
cv2.waitKey(0)
cv2.destroyAllWindows()

# Capture 200 images in a loop
for i in range(200):
    ret, frame = video_capture.read()  # Capture a frame from camera
    frame = cv2.flip(frame, 1, 0)  # Flip frame to show mirror image
    print(i)  # Print iteration number to know how many images captured so far

    # Save image to the directory and change name for each iteration
    cv2.imwrite(new_path + "/" + name + '_' + str(i) + ".jpg", frame)
    time.sleep(0.1)  # Add delay of 0.1 sec between captures

print('DONE')  # Print "DONE" to notify user that image capture is complete
video_capture.release()  # Release the camera