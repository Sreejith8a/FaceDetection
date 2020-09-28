import cv2
from random import randrange
# Loading some pre-trained frontal-face data from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an video to detect faces in
webcam = cv2.VideoCapture(0)  # 0 refers default webcamera and videofiles can also be named 
print("Press Esc Key to quit")
while True:

    #read the current frame
    successful_frame_read, frame = webcam.read()

    # convert to gray-scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # Draw the rectangular on detection
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(256),randrange(256),randrange(256)),3)

    # show the frame
    cv2.imshow('Video Face Detection', frame)
    key = cv2.waitKey(1)       # 1 millisec between each frame
    
    if key==27:
        print("Quitting...")
        break


webcam.release()
print("No issues")