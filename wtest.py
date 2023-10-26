import cv2

def find_available_cameras(max_cameras_to_check=10):
    available_cameras = []

    for index in range(max_cameras_to_check):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()

    return available_cameras

if __name__ == "__main__":
    cameras = find_available_cameras()
    if cameras:
        print(f"Found available cameras at indices: {cameras}")
    else:
        print("No available cameras found.")
