import cv2
from realsense_configure import *
from ultralytics import YOLO
import math
import numpy as np

# Configure Yolo Model
model = YOLO("train18/weights/best.pt")

# object classes
classNames = ['BigBox', 'Nozzle', 'Rocket', 'SmallBox']
classColors = { 'BigBox' : (235, 82, 52),
                'Nozzle' : (235, 217, 52),
                'Rocket' : (52, 235, 73),
                'SmallBox' : (230, 46, 208)}

def get_vals(depth_image, color_image):
    cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results = model(color_image, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:

            confidence = math.ceil((box.conf[0]*100))/100
            if confidence > 0.4:
                cls = int(box.cls[0])
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # Calculate Depth
                # Find Center of Bounding Box
                centerx = int((x2 + x1)/2)
                centery = int((y2 + y1)/2)
                bx1, bx2 = sorted([x1, x2])
                by1, by2 = sorted([y1, y2])
                className = classNames[cls]
                classColor = classColors[className]
                distance = depth_frame[centery, centerx]
                distanceinches = (depth_frame[centery, centerx])/25.2

                depth_list = []
                # Now iterate through the bounded box
                for i in range(by1, by2):  # rows
                    for j in range(bx1, bx2):  # columns
                        depth_value = depth_frame[i, j]

                        depth_list.append(depth_value)

                if len(depth_list) > 0:
                    depth_value = sum(depth_list)/len(depth_list)
                    depth = (depth_value)/25.2
                else:
                    depth_value = 0
                    depth = 0



                cv2.rectangle(color_image, (x1, y1), (x2, y2), classColor, 3)

                # confidence
                print("Confidence --->",confidence)

                # class name
                print("Class name -->", className)


                print(f"Distance in ---> {distanceinches:.3f} in",)
                print(f"Distance ---> {distance:.3f} mm",)
                print(f"Av BBox Distance in ---> {depth:.3f} in",)
                print(f"Av BBox Distance ---> {depth_value:.3f} mm",)
                # object details
                org = [x1, y1]
                bottom = [x1, y2+3]
                bbottom = [x1, y2+35]
                font = cv2.FONT_HERSHEY_DUPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(color_image, f"{className}: {confidence}", org, font, fontScale, color, thickness)

                cv2.putText(color_image, f"Distance: {distanceinches:.3f} in", bottom, font, fontScale, color, thickness)
                cv2.putText(color_image, f"Distance: {distance:.3f} mm", bbottom, font, fontScale, color, thickness)




def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


if __name__ == "__main__":
    print("[INFO] starting video stream...")

    # Initialize Camera Intel Realsense
    dc = DepthCamera()

    # Leave Space here to Configure Settings
    dc.set_Settings_from_json('camerasettings/test.json')
    # Start Camera
    print("[INFO] starting video stream...")
    dc.start_Streaming()
    # Create mouse event
    cv2.namedWindow("Color frame")

    while True:
        ret, depth_frame, color_frame = dc.get_frame()

        cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        results = model(color_frame, stream=True)

        get_vals(depth_frame, color_frame)
        # Show distance for a specific point


        cv2.imshow("depth frame", depth_frame)
        cv2.imshow("Color frame", color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
