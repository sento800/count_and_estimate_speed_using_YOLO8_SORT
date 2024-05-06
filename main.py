from ultralytics import YOLO
import cv2
import cvzone
import math
from estimateSpeed import estimateSpeed
from sort import *

cap = cv2.VideoCapture("./video/cars4.mp4")  # For Video

model = YOLO("./Yolo-Weights/yolov8l.pt")
name = model.model.names

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


limits = [300, 297, 1000, 297]
area1 = [(500, 100), (800, 100),(800,130),(500,130)]
area2 = [(200, 400), (1200, 400),(1200,500),(200,500)]
totalCount = []
vehicles_entering = {}

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    results = model(img, stream=True)
    # fps = int(cap.get(cv2.CAP_PROP_FPS))
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus"  and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))
                cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=2, offset=5)


    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    cv2.polylines(img, [np.array(area2,np.int32)],True,(0,0,255),6)
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        cx, cy = x1 + w // 2, y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=5, rt=1, colorR=(255, 0, 255))
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        cvzone.putTextRect(img, f'{id}q', (max(0, x1), max(60, y1)),
                                   scale=1, thickness=2, offset=5)
        resultPoint1 = cv2.pointPolygonTest(np.array(area1, np.int32), (int(cx), int(cy)), False)
        if resultPoint1 >= 0:
            vehicles_entering[id] = [[x1, y1, x2, y2]]

        if id in vehicles_entering:
            resultPoint2 = cv2.pointPolygonTest(np.array(area2, np.int32), (int(cx), int(cy)), False)
            if resultPoint2>=0:
                if len(vehicles_entering[id]) < 2:
                    vehicles_entering[id].append([x1, y1, x2, y2])
                # elapsed_time2 = vehicles_entering[id][1]-vehicles_entering[id][0]
                # distance2 = 10
                # speed_ms2 = distance2 / elapsed_time2
                # speed_km2 = speed_ms2 * 3.6 * 15
                speed_km2 = estimateSpeed(vehicles_entering[id][0],vehicles_entering[id][1],fps,w)
                cvzone.putTextRect(img, f' {int(speed_km2)} km/h', (max(0, x2), max(35, y2)), scale=1, thickness=1, offset=5)
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cvzone.putTextRect(img, f'car: {int(len(totalCount))}', (255, 100), scale=2, thickness=3, offset=10)
    cv2.putText(img,f'fps : {int(fps)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,225),2)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


