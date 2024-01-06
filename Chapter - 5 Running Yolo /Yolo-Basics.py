from ultralytics import YOLO
import cv2

model = YOLO('yolov6-n.pt')
results = model('Images/1.png',show=True)
cv2.waitKey(0)
