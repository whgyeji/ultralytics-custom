from ultralytics import YOLOE
import cv2
# Initialize a YOLOE model
# model = YOLOE("yoloe-26l-seg.pt")  # or yoloe-26s/m-seg.pt for different sizes
# model = YOLOE("yoloe-v8m-seg.pt")
# model = YOLOE("runs/detect/train13/weights/last.pt")
# model = YOLOE("runs/detect/train27/weights/last.pt")
# model = YOLOE("runs/detect/train72/weights/last.pt")
model = YOLOE("runs/detect/train86/weights/last.pt")

# model = YOLOE("yoloe-v8l-seg-pf.pt")

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
model.set_classes(["road_obstacle"])
# model.set_classes(["road_obstacle", "car", "person", "motorcycle", "bus", "truck"])


# Run detection on the given image
# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260331/images3', save=True)
# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260331/images_crop', save=True)
# results = model.predict('/home/data/wanghonggang/dataset/drive_test/renmo_test/images', save=True)
# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260422/images', save=True)
results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260422_crop/aa', save=True)
# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260422_crop/images_crop', save=True)


# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/box_test/', save=True)
# results = model.predict('/home/data/wanghonggang/whg_test/yoloworld_test/20260418/tmp', save=True)

# results = model.predict('/home/data/wanghonggang/dataset/drive_test/shuima/images', save=True)


# results = model.predict('ultralytics/assets/bus.jpg', save=True)
# results[0].save('result.jpg')
