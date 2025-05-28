import cv2
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# 강조 함수 (붉은 테두리 + 밝기 강조)
def highlight_objects(img, boxes, indexes):
    for i in range(len(boxes)):
        if i in indexes:
            if boxes[i] is None or len(boxes[i]) != 4:
                continue

            x, y, w, h = boxes[i]
            x, y, w, h = int(x), int(y), int(w), int(h)

            # 이미지 범위 확인
            if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
                continue

            sub_img = img[y:y+h, x:x+w]
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            img[y:y+h, x:x+w] = res

            # 붉은 테두리
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    return img

# 객체 감지 함수
def detect_objects(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if indexes is not None and len(indexes) > 0:
        indexes = indexes.flatten()
    else:
        indexes = []

    return boxes, confidences, class_ids, indexes

# 카메라 실행
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, confidences, class_ids, indexes = detect_objects(frame)
    frame = highlight_objects(frame, boxes, indexes)

    # 클래스명 라벨 출력
    for i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv3 Object Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC 키로 종료
        break

cap.release()
cv2.destroyAllWindows()
