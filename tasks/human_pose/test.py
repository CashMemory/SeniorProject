import cv2

cap = cv2.VideoCapture(0)



try:
    while(True):
        ret, frame = cap.read()

        cv2.imshow('Frames', frame)



except:
	print("Video has ended")
