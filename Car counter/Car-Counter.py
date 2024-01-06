from ultralytics import YOLO
import cv2
import cvzone
import Name
import math
from sort import *

cap = cv2.VideoCapture('Cvideo.mp4')
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('../Yolo-weights/yolov8n.pt')

classNames = Name.CocoName()

mask = cv2.imread("mask.png")

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

limits = [1600, 620, 500, 620]

totalCount = []

while True:
    success, img = cap.read()
    imageRegion = cv2.bitwise_and(img,mask)

    results = model(img, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1,y1), (x2,y2),(255,0,255),3)
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])

            currentclass = classNames[cls]

            if currentclass == 'car' and conf > 0.3:
                #cvzone.putTextRect(img,f'{currentclass} {conf}', (max(0, x1), max(35, y1)), scale=2, thickness=2)
                cvzone.cornerRect(img, (x1, y1, w, h))
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255), 10)

    for result in resultTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=2)

        cx,cy = x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),10,(255,0,255), cv2.FILLED)

        if limits[0] < cx < limits[2] or limits[1] - 20 < cy < limits[1] + 20:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 10)

    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))

    cv2.imshow("Image", img)
    #cv2.imshow("imageRegion", imageRegion)
    cv2.waitKey(1)




