from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='vt/data.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

results = model("camera_5.jpg")  # predict on an image

# Export the model to ONNX format
success = model.export(format='onnx')