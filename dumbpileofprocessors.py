from ultralytics import YOLO



# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')  # build from YAML and transfer weights

# Train the model for 3 epochs
results = model.train(data='BlenderModel/data.yaml', batch=4, epochs=3, plots=True)

# Evaluate the model's performance on the validation set
results1 = model.val()

results2 = model("img.png")  # predict on an image

# Export the model to ONNX format
success = model.export(format='onnx')
