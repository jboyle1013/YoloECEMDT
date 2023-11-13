from ultralytics import YOLO
import cv2
import math
import time
import pyrealsense2 as rs
from imutils.video import FPS
import dill




# model
model = YOLO("segtrain2/weights/best.pt")

# object classes
classNames = ['BigBox', 'Nozzle', 'Rocket', 'SmallBox']
classColors = { 'BigBox' : (235, 82, 52),
                'Nozzle' : (235, 217, 52),
                'Rocket' : (52, 235, 73),
                'SmallBox' : (230, 46, 208)}

# start webcam
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(3)

# FPS: used to compute the (approximate) frames per second
# Start the FPS timer
fps = FPS().start()

# loop over the frames from the video stream
cap.set(3, 640)
cap.set(4, 480)


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:

            confidence = math.ceil((box.conf[0]*100))/100
            if confidence > 0.6:
                cls = int(box.cls[0])
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                className = classNames[cls]
                classColor = classColors[className]
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), classColor, 3)

                # confidence
                print(f"Confidence ---> {confidence*100}%")

                # class name

                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = .75
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, f"{className}: {confidence*100}*", org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()