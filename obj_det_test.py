# this is a test file for practicing object detection
import cv2 as cv


cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("error opening camera")
    exit()

while True:
    # capture frame by frame
    ret, frame = cap.read()
    if not ret:
        print("cannot read frame–stream may have ended")
        