import cv2
import numpy as np

# 전역 변수 초기화
drawing = False
ix, iy = -1, -1
rx1, ry1, rx2, ry2 = -1, -1, -1, -1

# 마우스 콜백 함수
def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, rx1, ry1, rx2, ry2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rx1, ry1, rx2, ry2 = ix, iy, x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rx1, ry1 = min(ix, x), min(iy, y)
        rx2, ry2 = max(ix, x), max(iy, y)

# 대비 조절 함수
def increase_contrast(img):
    return cv2.convertScaleAbs(img, alpha=2.0, beta=0)

# 드래그 영역 내에서만 * 표시
def draw_stars(img, rx1, ry1, rx2, ry2):
    output = img.copy()

    # 유효한 드래그 영역인지 확인
    if rx1 != -1 and ry1 != -1 and rx2 != -1 and ry2 != -1 and rx2 > rx1 and ry2 > ry1:
        for y in range(ry1, ry2, 10):
            for x in range(rx1, rx2, 10):
                b, g, r = img[y, x]
                if r > 1.5 * g and r > 1.5 * b and r > 100:
                    cv2.putText(output, '*', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return output

# 메인 실행
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_rectangle)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 별은 드래그 영역 안에서만
    frame_with_stars = draw_stars(frame, rx1, ry1, rx2, ry2)

    # 드래그 영역이 유효할 때만 필터 적용
    if rx1 != -1 and ry1 != -1 and rx2 != -1 and ry2 != -1 and rx2 > rx1 and ry2 > ry1:
        roi = frame_with_stars[ry1:ry2, rx1:rx2]
        contrasted_roi = increase_contrast(roi)
        frame_with_stars[ry1:ry2, rx1:rx2] = contrasted_roi
        cv2.rectangle(frame_with_stars, (rx1, ry1), (rx2, ry2), (0, 255, 0), 1)

    cv2.imshow('frame', frame_with_stars)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
