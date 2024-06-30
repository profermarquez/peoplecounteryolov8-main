import math
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        new_center_points = {object_id: self.center_points[object_id] for _, _, _, _, object_id in objects_bbs_ids}
        self.center_points = new_center_points
        return objects_bbs_ids

model = YOLO('yolov8s.pt')

# Áreas centradas que ocupan 1/10 del ancho de la pantalla y toda la altura
area1 = [(359, 0), (461, 0), (461, 500), (359, 500)]
area2 = [(661, 0), (763, 0), (763, 500), (661, 500)]

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
per = 0
tracker = Tracker()
tracked_people = {}
crossed = 0

def is_inside_area(x, y, area):
    result = cv2.pointPolygonTest(np.array(area, np.int32), (x, y), False)
    return result >= 0

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    if len(a) == 0:
        continue

    px = pd.DataFrame(a).astype("float")
    objects_rect = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if 'person' in c:
            per += 1
            objects_rect.append((x1, y1, x2 - x1, y2 - y1))

    objects_bbs_ids = tracker.update(objects_rect)

    for obj in objects_bbs_ids:
        x, y, w, h, id = obj
        cx = (x + x + w) // 2
        cy = (y + y + h) // 2
        head_center = (cx, y)  # Cambia a las coordenadas de la cabeza

        if id not in tracked_people:
            tracked_people[id] = {'center': head_center, 'area1': False, 'area2': False}

        tracked_people[id]['center'] = head_center

        if is_inside_area(head_center[0], head_center[1], area1):
            print("dentro area1")
            #print(tracked_people[id]['area1'])
            tracked_people[id]['area1'] = True
            #print(tracked_people[id]['area1'])
            #tracked_people[id]['area2'] = False  # No puede estar en ambas áreas al mismo tiempo
        if is_inside_area(head_center[0], head_center[1], area2):
            print("dentro area2")
            tracked_people[id]['area2'] = True
            print(tracked_people[id])
            print(tracked_people[id]['area2'])
            print(tracked_people[id]['area1'])
            if tracked_people[id]['area1'] and tracked_people[id]['area2']:
                crossed += 1
                tracked_people[id]['area1'] = False
                tracked_people[id]['area2'] = False

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '1', (410, 250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.polylines(frame, [np.array(area2, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, '2', (710, 250), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.putText(frame, f'Crossed: {crossed}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        print(count)
        break

cap.release()
print(f'Total people detected: {per}')
print(f'People crossed both areas: {crossed}')
cv2.destroyAllWindows()
