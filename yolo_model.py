import os

from ultralytics import YOLOv10

config_path = 'C:\\Users\\WongKianWai\\OneDrive - Geoactive Limited\\Desktop\\Traffic Light Detection\\config.yaml'

model =  YOLOv10.from_pretrained("jameslahm/yolov10n")

model.train(data=config_path, epochs=50, batch=32)