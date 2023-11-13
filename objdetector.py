import cv2
from realsense_configure import *
from ultralytics import YOLO
import math
import numpy as np

# The Camera is set on top of its box and is currently 2.5 in or approximately 65mm (63.5 mm) above the table

# Configure Yolo Model
model = YOLO("train2/weights/best.pt")

# object classes
classNames = ['BigBox', 'Nozzle', 'Rocket', 'SmallBox']
classColors = { 'BigBox' : (235, 82, 52),
                'Nozzle' : (235, 217, 52),
                'Rocket' : (52, 235, 73),
                'SmallBox' : (230, 46, 208)}
def measure_dimensions(points_3d):
    """
    Measures the dimensions of an object from its point cloud.

    Parameters:
        points_3d: NumPy array of 3D points.

    Returns:
        A tuple containing the length, width, and height of the object.
    """
    x_coords, y_coords, z_coords = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    length = np.max(x_coords) - np.min(x_coords)
    width = np.max(y_coords) - np.min(y_coords)
    height = np.max(z_coords) - np.min(z_coords)

    return length, width, height


def calculate_position(points_3d):
    """
    Calculates the position in space of an object from its point cloud.

    Parameters:
        points_3d: NumPy array of 3D points.

    Returns:
        A tuple containing the x, y, z coordinates of the object's position.
    """
    centroid = np.mean(points_3d, axis=0)
    return tuple(centroid)



def calculate_depth(depth_frame, centerx, centery, dc):
    distance = depth_frame[centery, centerx]
    distance_inches = distance / 25.4

    # Calculate angles and components
    x_angle = (centerx - dc.depth_intrinsics.ppx) / dc.depth_intrinsics.fx
    y_angle = (centery - dc.depth_intrinsics.ppy) / dc.depth_intrinsics.fy
    horizontal_component = distance * math.tan(x_angle)
    vertical_component = distance * math.tan(y_angle)

    return distance, distance_inches, horizontal_component, vertical_component

def adjust_bbox_for_decimation(bbox, scale):
    x1, y1, x2, y2 = bbox
    return [int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)]


def get_vals(depth_image, color_image, depth_frame):

    results = model(color_image, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        masks = r.masks
        for box in boxes:

            confidence = math.ceil((box.conf[0]*100))/100
            if confidence > 0.6:
                cls = int(box.cls[0])
                # bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                bbox = [x1, y1, x2, y2]
                #adjusted_bbox = adjust_bbox_for_decimation(bbox, 2)
                points_3d = dc.get_3d_coordinates(depth_frame, bbox)
                # Calculate Depth
                # Find Center of Bounding Box
                centerx = int((x2 + x1)/2)
                centery = int((y2 + y1)/2)


                bx1, bx2 = sorted([x1, x2])
                by1, by2 = sorted([y1, y2])
                className = classNames[cls]
                classColor = classColors[className]

                bx1, bx2 = sorted([x1, x2])

                depth_list = []
                # Now iterate through the bounded box
                for i in range(by1, by2):  # rows
                    for j in range(bx1, bx2):  # columns
                        depth_value = depth_image[i, j]
                        depth_list.append(depth_value)

                depth_value = sum(depth_list)/len(depth_list)
                depth = (depth_value)
                depth_in = depth/25.4

                distance = depth_image[centery, centerx]
                distanceinches = (depth_image[centery, centerx])/25.4
                w, h = dc.depth_intrinsics.width, dc.depth_intrinsics.height
                ppx, ppy = dc.depth_intrinsics.ppx, dc.depth_intrinsics.ppy
                fx, fy = dc.depth_intrinsics.fx, dc.depth_intrinsics.fy


                x_angle = (centerx - ppx) / fx
                y_angle = (centery - ppy) / fy

                horizontal_component = distance * math.tan(x_angle)
                vertical_component = distance * math.tan(y_angle)
                calculated_distance = math.sqrt(pow(abs(horizontal_component), 2) + pow(abs(vertical_component), 2))
                dimensions = measure_dimensions(points_3d)
                position = calculate_position(points_3d)

                horizontal_angle, calcdistance, direction, forward_distance, horizontal_distance = dc.convert_to_direction_and_distance(*position)



                cv2.rectangle(color_image, (x1, y1), (x2, y2), classColor, 3)


                # confidence
                print(f"Confidence ---> {confidence*100:.1f}%")

                # class name
                print("Class name -->", className)

                print(f"Object Dimensions (Length x Width x Height): {dimensions}")
                print(f"Object Position (X, Y, Z): {position}")

                print(f"Distance in ---> {distanceinches:.3f} in",)
                print(f"Distance ---> {distance:.3f} mm",)
                print(f"Av Distance in ---> {depth_in:.3f} in",)
                print(f"Av Distance ---> {depth:.3f} mm",)
                print(f"Y angle  ---> {y_angle:.3f} rad",)
                print(f"X angle  ---> {x_angle:.3f} rad",)
                print(f"Calculated Vertical Distance mm ---> {vertical_component:.3f} mm",)
                print(f"Calculated Horizontal Distance mm ---> {horizontal_component:.3f} mm",)
                print(f"Calculated Hypotenuse mm ---> {calculated_distance:.3f} mm\n",)
                print("PointCloud Data")
                print(f"Angle: {horizontal_angle} degrees, Distance: {calcdistance} mm, Direction: {direction}")
                print(f"Horizontal Distance: {horizontal_distance} mm, Forward Distance: {forward_distance} mm")
                print(f"<----------------------------------------------------->\n")

                # object details
                org = [x1, y1]
                bottom = [x1, y2+3]
                bbottom = [x1, y2+35]
                font = cv2.FONT_HERSHEY_DUPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(color_image, f"{className}: {confidence*100:.0f}%", org, font, fontScale, color, thickness)

                cv2.putText(color_image, f"Distance: {distanceinches:.3f} in", bottom, font, fontScale, color, thickness)
                cv2.putText(color_image, f"Distance: {distance:.3f} mm", bbottom, font, fontScale, color, thickness)




def show_distance(event, x, y, args, params):
    global point
    point = (x, y)


if __name__ == "__main__":
    print("[INFO] Configuring Camera...")

    # Initialize Camera Intel Realsense
    dc = DepthCamera()

    # Leave Space here to Configure Settings
    dc.set_Settings_from_json('camerasettings/settings1.json')
    # Start Camera
    print("[INFO] starting video stream...")
    dc.start_Streaming()
    # Create mouse event
    cv2.namedWindow("Color frame")

    while True:
        ret, depth_image, color_frame, depth_colormap, depth_frame = dc.get_frame()

        #get_vals(depth_image, color_frame, depth_frame)
        # Show distance for a specific point

        cv2.imshow("depth frame", depth_colormap)
        cv2.imshow("Color frame", color_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
