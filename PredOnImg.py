import os

from PIL import Image
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("segtrain2/weights/best.pt")

# object classes
classNames = ['SmallBox', 'Rocket', 'BigBox', 'Nozzle']
for _, image in enumerate(os.listdir("images")):
    # Run inference on 'bus.jpg'
    results = model(f"images/{image}")  # results list

    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        im.save(f'segtrain2/res/results{_+1}.jpg')  # save image