import cv2
import numpy as np

# 색상 이름 사전 (단순화 버전)
def get_color_name(rgb):
    r, g, b = rgb
    if r > 200 and g < 100 and b < 100:
        return "Red"
    elif r < 100 and g > 200 and b < 100:
        return "Green"
    elif r < 100 and g < 100 and b > 200:
        return "Blue"
    elif r > 200 and g > 200 and b < 100:
        return "Yellow"
    elif r > 200 and g > 150 and b > 200:
        return "Pink"
    elif r > 200 and g > 200 and b > 200:
        return "White"
    elif r < 50 and g < 50 and b < 50:
        return "Black"
    else:
        return "Other"

# 색각 이상 시뮬레이션 행렬 (Protanopia)
M_protan = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281,  0.099216],
    [-0.003882, -0.048116, 1.051998]
])

def simulate_cvd(frame, matrix):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = rgb.reshape(-1, 3)
    simulated = np.clip(reshaped @ matrix.T, 0, 1)
    result = (simulated.reshape(frame.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

# 영상 처리 시작
cap = cv2.VideoCapture(0)
simulate = False  # 시뮬레이션 ON/OFF

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    # 중앙 픽셀 색상 추출
    b, g, r = frame[center_y, center_x]
    rgb = (r, g, b)
    hex_color = "#{:02X}{:02X}{:02X}".format(r, g, b)
    color_name = get_color_name(rgb)

    # 색각 이상 시뮬레이션 적용
    display_frame = simulate_cvd(frame, M_protan) if simulate else frame.copy()

    # 십자선 표시
    cv2.drawMarker(display_frame, (center_x, center_y), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

    # 색상 정보 텍스트 출력
    cv2.rectangle(display_frame, (10, 10), (300, 100), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Color: {color_name}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"RGB: {rgb}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 색상 바
    # 색상 바 그리기
    cv2.rectangle(display_frame, (20, h - 70), (300, h - 40), tuple(map(int, rgb[::-1])), -1)

    # 스펙트럼 바
    spectrum = np.zeros((30, 256, 3), dtype=np.uint8)
    for i in range(256):
        hsv_color = np.uint8([[[i, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        spectrum[:, i] = bgr_color
    display_frame[h - 30:h, 0:256] = spectrum

    # 강조 표시 (색상 위치 강조)
    hsv_val = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0][0]
    cv2.line(display_frame, (hsv_val, h - 30), (hsv_val, h), (255, 255, 255), 2)

    # 창 표시
    cv2.imshow("Color Detector (Press 's' to toggle simulation, 'ESC' to exit)", display_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        simulate = not simulate

cap.release()
cv2.destroyAllWindows()
