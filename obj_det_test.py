# this is a python project for practicing object detection
import cv2 as cv
from ultralytics import YOLO

# define model and
model = YOLO("yolo-Weights/yolov8n.pt")
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# create capture object
cap = cv.VideoCapture(0)

# catch error in opening camera
if not cap.isOpened():
    print("error opening camera")
    exit()

# loop to continuously capture frames from webcam
while True:
    # capture current frame
    ret, frame = cap.read()

    # catch error reading frame
    if not ret:
        print("cannot read frame–error or stream may have ended")
        break

    # send frame to model
    frame = cv.flip(frame, 1)
    result = model(frame, stream=True)

    # show frame (this is a video so wait 1ms)
    cv.imshow("webcam", frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()