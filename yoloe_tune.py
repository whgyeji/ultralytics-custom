from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEPETrainer, YOLOETrainerFromScratch

# # Initialize a detection model from a config
# model = YOLOE("yoloe-v8m.yaml")

# # Load weights from a pretrained segmentation checkpoint (same scale)
# # model.load("yoloe-v8m-seg.pt")
# model.load("runs/detect/train46/weights/last.pt")



model = YOLOE("yoloe-v8s.yaml")

# Load weights from a pretrained segmentation checkpoint (same scale)
model.load("yoloe-v8s-seg.pt")

# # Train only the classification branch
# results = model.train(

#     # data="adas.yaml",  # Detection dataset
#     # data="coco128.yaml",  # Detection dataset
#     data="adas_json.yaml",  # Detection dataset
#     epochs=80,
#     patience=10,
#     trainer=YOLOEPETrainer,  # <- Important: use detection trainer
# )

model.train(
    # data='adas.yaml',
    data='adas_obstacle.yaml',
    epochs=80,
    patience=50,
    trainer=YOLOEPETrainer,  # <- Important: use detection trainer
    # mosaic=0.0,                    # 建议关闭 Mosaic，避免冲突
    # mixup=0.0                      # 建议关闭 MixUp
)