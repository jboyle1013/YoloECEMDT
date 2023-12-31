## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
import math

###############################################
##      Open CV and Numpy integration        ##
###############################################
from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2

# Configure Yolo Model
model = YOLO("train13/weights/best.pt")

# object classes
classNames = ['BigBox', 'Nozzle', 'Rocket', 'SmallBox']

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
print("Starting Stream")
# Start streaming
pipeline.start(config)

try:
    while True:


        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = model(color_image, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:

                confidence = math.ceil((box.conf[0]*100))/100
                if confidence > 0.5:

                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                    bx1, bx2 = sorted([x1, x2])
                    by1, by2 = sorted([y1, y2])
                    # Calculate Depth
                    # Find Center of Bounding Box
                    centerx = int((x2 + x1)/2)
                    centery = int((y2 + y1)/2)

                    depth_list = []
                    # Now iterate through the bounded box
                    for i in range(by1, by2):  # rows
                        for j in range(bx1, bx2):  # columns
                            depth_value = depth_frame[j, i]
                            depth_list.append(depth_value)

                    depth_value = sum(depth_list)/len(depth_list)
                    depth = (depth_value )
                    # put box in cam
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])


                    print(f"Distance ---> {depth:.3f} in",)
                    # object details
                    org = [x1, y1]
                    bottom = [x1, y2+3]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(color_image, f"{classNames[cls]}: {confidence}", org, font, fontScale, color, thickness)

                    cv2.putText(color_image, f"Distance: {depth:.3f} in", bottom, font, fontScale, color, thickness)

            # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()