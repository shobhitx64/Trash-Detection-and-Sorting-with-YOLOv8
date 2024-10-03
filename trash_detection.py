from ultralytics import YOLO
import torch
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

'''
model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    train_results = model.train(
        data="C:\\Users\\shobhit\\Downloads\\handProject\\trash2\\data.yaml",
        epochs=500,
        imgsz=416,
        device="cuda:0",
        optimizer='Adam',
        patience=10,
        save_period=50
    )

    print(f"saved at {model.best}")
'''


model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))
tracker = DeepSort(max_age=3)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

bin_colors = {
    'Aluminium foil': 'Blue',
    'Bottle cap': 'Blue',
    'Bottle': 'Blue',
    'Broken glass': 'Black',
    'Can': 'Blue',
    'Carton': 'Blue',
    'Cigarette': 'Black',
    'Cup': 'Green',
    'Lid': 'Black',
    'Other litter': 'Black',
    'Other plastic': 'Blue',
    'Paper': 'Green',
    'Plastic bag - wrapper': 'Black',
    'Plastic container': 'Blue',
    'Pop tab': 'Blue',
    'Straw': 'Black',
    'Styrofoam piece': 'Black',
    'Unlabeled litter': 'Black'
}


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)
    color = (0, 0, 0)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            detections.append(([x1, y1, x2-x1, y2-y1], confidence, class_id))           
            
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        
        x1, y1, x2, y2 = map(int, track.to_tlbr())
        class_id = track.det_class
        class_name = model.names[class_id]
        correct_bin = bin_colors.get(class_name)
        
        if correct_bin == 'Blue':
            color = (255, 0, 0) 
        elif correct_bin == 'Black':
            color = (0, 0, 0)
        else:
            color = (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'{class_name} || {correct_bin}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)

    cv2.imshow('Trash Sorter', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
