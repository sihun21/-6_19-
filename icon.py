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
    elif r > 200 and g > 150 and b > 200:  # 분홍색을 위해 조정
        return "분홍색"
    elif r > 200 and g > 200 and b > 200:  # 흰색을 위해 조정
        return "흰색"
    elif r < 50 and g < 50 and b < 50:  # 검은색을 위해 조정
        return "검은색"
    else:
        return "기타"


M_protan = np.array([
    [0.152286, 1.052583, -0.204868],
    [0.114503, 0.786281, 0.099216],
    [-0.003882, -0.048116, 1.051998]
])


def simulate_cvd(frame, matrix):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = rgb_frame.reshape(-1, 3)
    simulated = np.clip(reshaped @ matrix.T, 0, 1)
    result = (simulated.reshape(frame.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


# --- 새로운 기능: 버튼 이미지 로드 및 UI 요소 정의 ---
icon_size = (50, 50)  # 아이콘 너비, 높이

# 빨간색 버튼 이미지 로드
try:
    red_button_img = cv2.imread('red butten.png')  # 'red butten.png' 파일이 같은 디렉토리에 있는지 확인하세요.
    if red_button_img is None:
        print(f"경고: 'red butten.png'를 None으로 로드했습니다. 경로와 파일 무결성을 확인하세요.")
        raise FileNotFoundError
    red_button_icon = cv2.resize(red_button_img, icon_size)
except FileNotFoundError:
    print("경고: 'red butten.png' 파일을 찾을 수 없거나 로드에 실패했습니다. 플레이스홀더를 사용합니다.")
    red_button_icon = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
    red_button_icon[:] = (0, 0, 200)  # 어두운 빨간색 (BGR)
    cv2.putText(red_button_icon, "R", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 파란색 버튼 이미지 로드
try:
    blue_button_img = cv2.imread('blue butten.png')  # 'blue butten.png' 파일이 같은 디렉토리에 있는지 확인하세요.
    if blue_button_img is None:
        print(f"경고: 'blue butten.png'를 None으로 로드했습니다. 경로와 파일 무결성을 확인하세요.")
        raise FileNotFoundError
    blue_button_icon = cv2.resize(blue_button_img, icon_size)
except FileNotFoundError:
    print("경고: 'blue butten.png' 파일을 찾을 수 없거나 로드에 실패했습니다. 플레이스홀더를 사용합니다.")
    blue_button_icon = np.zeros((icon_size[1], icon_size[0], 3), dtype=np.uint8)
    blue_button_icon[:] = (200, 0, 0)  # 어두운 파란색 (BGR)
    cv2.putText(blue_button_icon, "B", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# 버튼 위치 (x, y, 너비, 높이)
text_info_height = 100
button_margin = 10
red_button_rect = {'x': 10, 'y': text_info_height + button_margin, 'w': icon_size[0], 'h': icon_size[1]}
blue_button_rect = {'x': red_button_rect['x'] + icon_size[0] + button_margin, 'y': text_info_height + button_margin,
                    'w': icon_size[0], 'h': icon_size[1]}

# 필터 상태
active_color_filter = None
filter_alpha = 0.3


# --- 마우스 클릭 처리를 위한 콜백 함수 ---
def handle_mouse_click(event, x, y, flags, param):
    global active_color_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        if red_button_rect['x'] <= x <= red_button_rect['x'] + red_button_rect['w'] and \
                red_button_rect['y'] <= y <= red_button_rect['y'] + red_button_rect['h']:
            active_color_filter = "red" if active_color_filter != "red" else None
            print(f"빨간색 필터 {'활성화됨' if active_color_filter == 'red' else '비활성화됨'}.")
        elif blue_button_rect['x'] <= x <= blue_button_rect['x'] + blue_button_rect['w'] and \
                blue_button_rect['y'] <= y <= blue_button_rect['y'] + blue_button_rect['h']:
            active_color_filter = "blue" if active_color_filter != "blue" else None
            print(f"파란색 필터 {'활성화됨' if active_color_filter == 'blue' else '비활성화됨'}.")


# --- 비디오 처리 시작 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 비디오 스트림을 열 수 없습니다.")
    exit()

simulate_cvd_active = False

window_title = "색상 감지기 ('s': 시뮬레이션 토글, 'ESC': 종료)"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, handle_mouse_click)

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

    # 2. 색각 이상 시뮬레이션 적용 (활성화된 경우)
    if simulate_cvd_active:
        processed_frame = simulate_cvd(frame, M_protan)
    else:
        processed_frame = frame.copy()

    # 3. 표시할 프레임 준비
    display_frame = processed_frame.copy()

    # 4. 활성 색상 필터 적용
    if active_color_filter == "red":
        overlay = np.full(display_frame.shape, (0, 0, 255), dtype=np.uint8)
        cv2.addWeighted(overlay, filter_alpha, display_frame, 1 - filter_alpha, 0, display_frame)
    elif active_color_filter == "blue":
        overlay = np.full(display_frame.shape, (255, 0, 0), dtype=np.uint8)
        cv2.addWeighted(overlay, filter_alpha, display_frame, 1 - filter_alpha, 0, display_frame)

    # 5. UI 요소 그리기
    display_frame[red_button_rect['y']:red_button_rect['y'] + red_button_rect['h'],
    red_button_rect['x']:red_button_rect['x'] + red_button_rect['w']] = red_button_icon
    display_frame[blue_button_rect['y']:blue_button_rect['y'] + blue_button_rect['h'],
    blue_button_rect['x']:blue_button_rect['x'] + blue_button_rect['w']] = blue_button_icon

    cv2.drawMarker(display_frame, (center_x, center_y), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

    cv2.rectangle(display_frame, (10, 10), (300, text_info_height), (0, 0, 0), -1)
    cv2.putText(display_frame, f"색상: {color_name_at_center}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                2)
    cv2.putText(display_frame, f"RGB: {rgb_at_center}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(display_frame, (20, h - 70), (300, h - 40), tuple(map(int, rgb_at_center[::-1])), -1)

    spectrum_bar_height = 30
    spectrum_bar_width = 256
    spectrum = np.zeros((spectrum_bar_height, spectrum_bar_width, 3), dtype=np.uint8)
    for i in range(spectrum_bar_width):
        hsv_color = np.uint8([[[i * 180 // spectrum_bar_width, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        spectrum[:, i] = bgr_color
    display_frame[h - spectrum_bar_height:h, 0:spectrum_bar_width] = spectrum

    # 스펙트럼 막대에 하이라이트 그리기 (OverflowError 수정 적용됨)
    hsv_original_center_color = cv2.cvtColor(np.uint8([[rgb_at_center[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue_val_from_cv = hsv_original_center_color[0]

    # 방어적 코딩: cvtColor에서 비정상적인 값이 나올 경우를 대비해 hue 값을 OpenCV 표준 범위(0-179)로 제한합니다.
    clipped_hue = np.clip(hue_val_from_cv, 0, 179)

    # 제한된 hue 값을 사용하여 위치를 계산합니다.
    hue_pos_float = float(clipped_hue) * spectrum_bar_width / 180.0
    hue_pos_on_bar = int(hue_pos_float)

    # 추가적인 안전장치: 계산된 위치가 스펙트럼 막대의 유효한 인덱스 범위 [0, spectrum_bar_width - 1] 내에 있도록 합니다.
    hue_pos_on_bar = max(0, min(hue_pos_on_bar, spectrum_bar_width - 1))

    cv2.line(display_frame, (hue_pos_on_bar, h - spectrum_bar_height), (hue_pos_on_bar, h), (255, 255, 255), 2)

    # 6. 최종 프레임 표시
    cv2.imshow(window_title, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break
    elif key == ord('s'):  # 's' 키
        simulate_cvd_active = not simulate_cvd_active
        print(f"색각 이상 시뮬레이션이 {'켜졌습니다' if simulate_cvd_active else '꺼졌습니다'}.")

cap.release()
cv2.destroyAllWindows()