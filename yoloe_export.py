from ultralytics import YOLOE

# Select yoloe-26s/m-seg.pt for different sizes
# model = YOLOE("yoloe-v8m-seg.pt")
# model = YOLOE("runs/detect/train4/weights/best.pt")
model = YOLOE("runs/detect/train75/weights/last.pt")
# model = YOLOE("yoloe-v8l-seg.pt")
model.info(
    detailed=False,
    verbose=True,
    imgsz=640
)
exit()

# Configure the set_classes() before exporting the model
model.set_classes(["road_obstacle"])

export_model = model.export(format="onnx", half=False)
model = YOLOE(export_model)

# # Run detection on the given image
# results = model.predict("path/to/image.jpg")

# # Show results
# results[0].show()