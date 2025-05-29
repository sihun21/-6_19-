import cv2
import numpy as np


# --- 사용자 제공 도우미 함수 (변경 없음) ---
def get_color_name(rgb):
    r, g, b = rgb
    if r > 200 and g < 100 and b < 100:
        return "빨간색"
    elif r < 100 and g > 200 and b < 100:
        return "초록색"
    elif r < 100 and g < 100 and b > 200:
        return "파란색"
    elif r > 200 and g > 200 and b < 100:
        return "노란색"
    elif r > 200 and g > 150 and b > 200:
        return "분홍색"
    elif r > 200 and g > 200 and b > 200:
        return "흰색"
    elif r < 50 and g < 50 and b < 50:
        return "검은색"
    else:
        return "기타"


# --- Machado (2009) 변환 행렬 (severity=1.0 기준) ---
CVD_MATRICES = {
    "protan": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]
    ]),
    "deutan": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881]
    ]),
    "tritan": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900]
    ])
}


def simulate_colorblind_machado(img, cb_type, severity=1.0):
    """Machado 시뮬레이션 + 강도 조절"""
    if cb_type not in CVD_MATRICES:
        return img
    matrix = CVD_MATRICES[cb_type]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = img_rgb.reshape(-1, 3)
    transformed = reshaped @ matrix.T
    transformed = np.clip(transformed, 0, 1)

    blended = (1 - severity) * reshaped + severity * transformed
    blended = np.clip(blended, 0, 1)
    result = (blended.reshape(img.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def daltonize(img, cb_type, severity=1.0, amplify=1.5):
    """Daltonization: 색약 보정"""
    if cb_type not in CVD_MATRICES:
        return img
    matrix = CVD_MATRICES[cb_type]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = img_rgb.reshape(-1, 3)

    simulated_initial = reshaped @ matrix.T
    simulated = (1 - severity) * reshaped + severity * simulated_initial
    simulated = np.clip(simulated, 0, 1)

    error = reshaped - simulated
    corrected = reshaped + amplify * error
    corrected = np.clip(corrected, 0, 1)

    result = (corrected.reshape(img.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# --- 새로운 기능: 버튼 이미지 로드 및 UI 요소 정의 ---
icon_size = (50, 50)  # 아이콘 너비, 높이

# 시뮬레이션 버튼 아이콘
# Protan (빨간색)
btn_icon_p = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_p[:] = (0, 0, 200)
cv2.putText(btn_icon_p, "P", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Deutan (초록색)
btn_icon_d = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_d[:] = (0, 200, 0)
cv2.putText(btn_icon_d, "D", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Tritan (파란색)
btn_icon_t = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_t[:] = (200, 0, 0)
cv2.putText(btn_icon_t, "T", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# 보정 (Daltonize) 버튼 아이콘
# Protan Daltonize (주황색)
btn_icon_pd = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_pd[:] = (0, 128, 255)  # BGR 주황색
cv2.putText(btn_icon_pd, "PD", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# Deutan Daltonize (청록색)
btn_icon_dd = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_dd[:] = (255, 255, 0)  # BGR 청록색
cv2.putText(btn_icon_dd, "DD", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

# Tritan Daltonize (자홍색)
btn_icon_td = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
btn_icon_td[:] = (255, 0, 255)  # BGR 자홍색
cv2.putText(btn_icon_td, "TD", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# 버튼 위치 정의
text_info_height = 100
button_margin = 10
button_row_height = icon_size[1] + button_margin

# 첫 번째 줄 버튼 (시뮬레이션)
btn_rect_p = {'x': 10, 'y': text_info_height + button_margin, 'w': icon_size[0], 'h': icon_size[1]}
btn_rect_d = {'x': btn_rect_p['x'] + icon_size[0] + button_margin, 'y': btn_rect_p['y'], 'w': icon_size[0],
              'h': icon_size[1]}
btn_rect_t = {'x': btn_rect_d['x'] + icon_size[0] + button_margin, 'y': btn_rect_p['y'], 'w': icon_size[0],
              'h': icon_size[1]}

# 두 번째 줄 버튼 (보정)
btn_rect_pd = {'x': 10, 'y': btn_rect_p['y'] + button_row_height, 'w': icon_size[0], 'h': icon_size[1]}
btn_rect_dd = {'x': btn_rect_pd['x'] + icon_size[0] + button_margin, 'y': btn_rect_pd['y'], 'w': icon_size[0],
               'h': icon_size[1]}
btn_rect_td = {'x': btn_rect_dd['x'] + icon_size[0] + button_margin, 'y': btn_rect_pd['y'], 'w': icon_size[0],
               'h': icon_size[1]}

# 현재 활성화된 모드 상태
active_cb_type = None  # "protan", "deutan", "tritan"
active_cb_mode = None  # "simulate", "correct"


# --- 마우스 클릭 처리를 위한 콜백 함수 ---
def handle_mouse_click(event, x, y, flags, param):
    global active_cb_type, active_cb_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        # 모든 모드를 초기화
        new_cb_type = None
        new_cb_mode = None

        # 시뮬레이션 버튼 체크
        if btn_rect_p['x'] <= x <= btn_rect_p['x'] + btn_rect_p['w'] and \
                btn_rect_p['y'] <= y <= btn_rect_p['y'] + btn_rect_p['h']:
            if active_cb_type == "protan" and active_cb_mode == "simulate":  # 이미 활성화된 경우 끔
                print("Protanopia 시뮬레이션 비활성화.")
            else:  # 새로 활성화
                new_cb_type = "protan"
                new_cb_mode = "simulate"
                print("Protanopia 시뮬레이션 활성화.")

        elif btn_rect_d['x'] <= x <= btn_rect_d['x'] + btn_rect_d['w'] and \
                btn_rect_d['y'] <= y <= btn_rect_d['y'] + btn_rect_d['h']:
            if active_cb_type == "deutan" and active_cb_mode == "simulate":
                print("Deuteranopia 시뮬레이션 비활성화.")
            else:
                new_cb_type = "deutan"
                new_cb_mode = "simulate"
                print("Deuteranopia 시뮬레이션 활성화.")

        elif btn_rect_t['x'] <= x <= btn_rect_t['x'] + btn_rect_t['w'] and \
                btn_rect_t['y'] <= y <= btn_rect_t['y'] + btn_rect_t['h']:
            if active_cb_type == "tritan" and active_cb_mode == "simulate":
                print("Tritanopia 시뮬레이션 비활성화.")
            else:
                new_cb_type = "tritan"
                new_cb_mode = "simulate"
                print("Tritanopia 시뮬레이션 활성화.")

        # 보정 (Daltonize) 버튼 체크
        elif btn_rect_pd['x'] <= x <= btn_rect_pd['x'] + btn_rect_pd['w'] and \
                btn_rect_pd['y'] <= y <= btn_rect_pd['y'] + btn_rect_pd['h']:
            if active_cb_type == "protan" and active_cb_mode == "correct":
                print("Protanopia 보정 비활성화.")
            else:
                new_cb_type = "protan"
                new_cb_mode = "correct"
                print("Protanopia 보정 활성화.")

        elif btn_rect_dd['x'] <= x <= btn_rect_dd['x'] + btn_rect_dd['w'] and \
                btn_rect_dd['y'] <= y <= btn_rect_dd['y'] + btn_rect_dd['h']:
            if active_cb_type == "deutan" and active_cb_mode == "correct":
                print("Deuteranopia 보정 비활성화.")
            else:
                new_cb_type = "deutan"
                new_cb_mode = "correct"
                print("Deuteranopia 보정 활성화.")

        elif btn_rect_td['x'] <= x <= btn_rect_td['x'] + btn_rect_td['w'] and \
                btn_rect_td['y'] <= y <= btn_rect_td['y'] + btn_rect_td['h']:
            if active_cb_type == "tritan" and active_cb_mode == "correct":
                print("Tritanopia 보정 비활성화.")
            else:
                new_cb_type = "tritan"
                new_cb_mode = "correct"
                print("Tritanopia 보정 활성화.")

        # 실제 활성 모드 업데이트: 클릭된 버튼이 현재 활성화된 모드와 같으면 끄고, 다르면 새 모드로 설정
        if new_cb_type == active_cb_type and new_cb_mode == active_cb_mode:
            active_cb_type = None
            active_cb_mode = None
        else:
            active_cb_type = new_cb_type
            active_cb_mode = new_cb_mode


# --- 비디오 처리 시작 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 비디오 스트림을 열 수 없습니다.")
    exit()

window_title = "색상 감지기 및 색약 시뮬레이션/보정 ('ESC': 종료)"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, handle_mouse_click)

# 강도 트랙바 추가 (0~200 값을 0.0~2.0으로 변환)
cv2.createTrackbar("Severity (x100)", window_title, 100, 200, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("오류: 프레임을 수신할 수 없습니다 (스트림 끝?). 종료합니다...")
        break

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    # 1. 원본 프레임 중앙 색상 정보 가져오기
    b_orig, g_orig, r_orig = frame[center_y, center_x]
    rgb_at_center = (r_orig, g_orig, b_orig)
    color_name_at_center = get_color_name(rgb_at_center)

    # 강도 트랙바 값 읽기
    severity_val = cv2.getTrackbarPos("Severity (x100)", window_title)
    severity = severity_val / 100.0

    # 2. 색각 이상 시뮬레이션 또는 보정 적용 (활성화된 경우)
    processed_frame = frame.copy()
    if active_cb_type and active_cb_mode:
        if active_cb_mode == "simulate":
            processed_frame = simulate_colorblind_machado(frame, active_cb_type, severity=severity)
        elif active_cb_mode == "correct":
            processed_frame = daltonize(frame, active_cb_type, severity=severity, amplify=1.5)

    # 3. 표시할 프레임 준비
    display_frame = processed_frame.copy()

    # 4. UI 요소 그리기
    # 시뮬레이션 버튼
    display_frame[btn_rect_p['y']:btn_rect_p['y'] + btn_rect_p['h'],
    btn_rect_p['x']:btn_rect_p['x'] + btn_rect_p['w']] = btn_icon_p
    display_frame[btn_rect_d['y']:btn_rect_d['y'] + btn_rect_d['h'],
    btn_rect_d['x']:btn_rect_d['x'] + btn_rect_d['w']] = btn_icon_d
    display_frame[btn_rect_t['y']:btn_rect_t['y'] + btn_rect_t['h'],
    btn_rect_t['x']:btn_rect_t['x'] + btn_rect_t['w']] = btn_icon_t

    # 보정 버튼
    display_frame[btn_rect_pd['y']:btn_rect_pd['y'] + btn_rect_pd['h'],
    btn_rect_pd['x']:btn_rect_pd['x'] + btn_rect_pd['w']] = btn_icon_pd
    display_frame[btn_rect_dd['y']:btn_rect_dd['y'] + btn_rect_dd['h'],
    btn_rect_dd['x']:btn_rect_dd['x'] + btn_rect_dd['w']] = btn_icon_dd
    display_frame[btn_rect_td['y']:btn_rect_td['y'] + btn_rect_td['h'],
    btn_rect_td['x']:btn_rect_td['x'] + btn_rect_td['w']] = btn_icon_td

    cv2.drawMarker(display_frame, (center_x, center_y), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

    # 상단 정보 박스
    cv2.rectangle(display_frame, (10, 10), (300, text_info_height), (0, 0, 0), -1)
    cv2.putText(display_frame, f"Color: {color_name_at_center}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255),
                2, cv2.LINE_AA)
    cv2.putText(display_frame, f"RGB: {rgb_at_center}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                cv2.LINE_AA)

    # 현재 활성 모드 및 강도 표시 (우측 상단)
    mode_display_text = "None"
    if active_cb_type and active_cb_mode:
        mode_display_text = f"{active_cb_type.capitalize()} {active_cb_mode.capitalize()}"

    cv2.putText(display_frame, f"Mode: {mode_display_text}", (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display_frame, f"Severity: {severity:.2f}", (w - 300, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                2, cv2.LINE_AA)

    cv2.rectangle(display_frame, (20, h - 70), (300, h - 40), tuple(map(int, rgb_at_center[::-1])), -1)

    spectrum_bar_height = 30
    spectrum_bar_width = 256
    spectrum = np.zeros((spectrum_bar_height, spectrum_bar_width, 3), dtype=np.uint8)
    for i in range(spectrum_bar_width):
        hsv_color = np.uint8([[[i * 180 // spectrum_bar_width, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        spectrum[:, i] = bgr_color
    display_frame[h - spectrum_bar_height:h, 0:spectrum_bar_width] = spectrum

    # 스펙트럼 막대에 하이라이트 그리기
    hsv_original_center_color = cv2.cvtColor(np.uint8([[rgb_at_center[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue_val_from_cv = hsv_original_center_color[0]

    clipped_hue = np.clip(hue_val_from_cv, 0, 179)
    hue_pos_float = float(clipped_hue) * spectrum_bar_width / 180.0
    hue_pos_on_bar = int(hue_pos_float)
    hue_pos_on_bar = max(0, min(hue_pos_on_bar, spectrum_bar_width - 1))

    cv2.line(display_frame, (hue_pos_on_bar, h - spectrum_bar_height), (hue_pos_on_bar, h), (255, 255, 255), 2)

    # 5. 최종 프레임 표시
    cv2.imshow(window_title, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()