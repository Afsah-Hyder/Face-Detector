import cv2
import numpy as np

cap=cv2.VideoCapture(0)            #which camera to be chosen
if cap.isOpened()==False:
    print("Can't open camera")
    exit()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:
    ret,frame=cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    img=cv2.flip(frame,1)         #1 for flip left-rigth, 0 for flip up down
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #for gray scale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xFF== ord('q'):             
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
