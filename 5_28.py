import cv2
import numpy as np

# 색상 이름 추출 (HSV 기반)
def get_color_name(rgb):
    r, g, b = rgb
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if v < 40:
        return "Black"
    if s < 40:
        return "White" if v > 200 else "Gray"
    if h < 15 or h >= 170:
        return "Red"
    elif 15 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 130:
        return "Blue"
    elif 130 <= h < 150:
        return "Navy"
    elif 150 <= h < 170:
        return "Purple"
    return "Other"

# 스펙트럼 색상 리스트 (BGR 순서)
gradient_colors = [
    (0, 0, 255),       # Red
    (0, 127, 255),     # Orange
    (0, 255, 255),     # Yellow
    (0, 255, 0),       # Green
    (255, 0, 0),       # Blue
    (130, 0, 75),      # Navy
    (211, 0, 148)      # Purple
]

# 색상 이름 → 인덱스 매핑
color_position_map = {
    "red": 0,
    "orange": 1,
    "yellow": 2,
    "green": 3,
    "blue": 4,
    "navy": 5,
    "purple": 6
}

# 스펙트럼 바 생성 함수
def create_rgb_gradient_bar(width, height, colors):
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    num_segments = len(colors) - 1
    seg_width = width // num_segments
    for i in range(num_segments):
        start = np.array(colors[i], dtype=np.float32)
        end = np.array(colors[i + 1], dtype=np.float32)
        for x in range(seg_width):
            t = x / seg_width
            color = ((1 - t) * start + t * end).astype(np.uint8)
            gradient[:, i * seg_width + x] = color
    if width % num_segments != 0:
        gradient[:, seg_width * num_segments:] = colors[-1]
    return gradient

# 카메라 시작
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # 중앙 색상 추출 및 이름
    b, g, r = frame[cy, cx]
    rgb = (r, g, b)
    color_name = get_color_name(rgb).lower()

    # UI 프레임 복사
    display = frame.copy()

    # 십자선
    cv2.drawMarker(display, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 40, 2)

    # 색상 바
    bar_w = 260
    bar_h = 20
    bar_x = w // 2 - bar_w // 2
    bar_y = h - 120
    cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 0, 0), -1)
    cv2.rectangle(display, (bar_x + 2, bar_y + 2), (bar_x + bar_w - 2, bar_y + bar_h - 2), tuple(map(int, rgb[::-1])), -1)

    # 색상 이름 표시
    text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)[0]
    text_x = (w - text_size[0]) // 2
    text_y = h - 50
    cv2.putText(display, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.putText(display, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 3, cv2.LINE_AA)

    # 스펙트럼 바
    spectrum_height = 30
    spectrum = create_rgb_gradient_bar(w, spectrum_height, gradient_colors)
    display[h - spectrum_height:h, 0:w] = spectrum

    # 현재 색상 위치에 화살표 표시
    if color_name in color_position_map:
        idx = color_position_map[color_name]
        num_segments = len(gradient_colors) - 1
        seg_width = w // num_segments
        center_x = idx * seg_width + seg_width // 2
        arrow_tip = (center_x, h - spectrum_height - 10)
        arrow_base_l = (center_x - 10, h - spectrum_height - 25)
        arrow_base_r = (center_x + 10, h - spectrum_height - 25)
        cv2.drawContours(display, [np.array([arrow_tip, arrow_base_l, arrow_base_r])], 0, (255, 255, 255), -1)

    # 출력
    cv2.imshow("Color UI", display)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
