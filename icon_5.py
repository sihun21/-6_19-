import cv2
import numpy as np
import matplotlib.colors as mcolors  # matplotlib.colors 모듈 임포트

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


# --- 사용자 제공 도우미 함수 (matplotlib.colors 사용) ---
# matplotlib의 CSS4_COLORS는 약 150여 가지의 색상 이름을 포함합니다.
# 더 많은 색상을 원하면 XKCD_COLORS를 사용할 수 있지만, 'xkcd:' 접두사가 붙습니다.
def get_color_name(rgb_tuple):
    # RGB 0-255 값을 0-1 범위로 정규화
    rgb_normalized = (rgb_tuple[0] / 255.0, rgb_tuple[1] / 255.0, rgb_tuple[2] / 255.0)

    min_distance = float('inf')
    closest_color_name = "Unknown Color"

    # matplotlib.colors.CSS4_COLORS 딕셔너리 사용
    for name, hex_value in mcolors.CSS4_COLORS.items():
        # 헥스 코드를 RGB (0-1) 튜플로 변환
        # mcolors.to_rgb는 헥스 코드나 색상 이름을 RGB (0-1)로 변환합니다.
        color_rgb_normalized = mcolors.to_rgb(hex_value)

        # 유클리드 거리의 제곱을 계산 (sqrt를 피하여 성능 향상)
        distance = sum([(c1 - c2) ** 2 for c1, c2 in zip(rgb_normalized, color_rgb_normalized)])

        if distance < min_distance:
            min_distance = distance
            closest_color_name = name

    # XKCD 색상 이름을 추가하려면 아래 주석을 해제하고 'xkcd:' 접두사를 처리해야 합니다.
    # for name, hex_value in mcolors.XKCD_COLORS.items():
    #     color_rgb_normalized = mcolors.to_rgb(hex_value)
    #     distance = sum([(c1 - c2)**2 for c1, c2 in zip(rgb_normalized, color_rgb_normalized)])
    #     if distance < min_distance:
    #         min_distance = distance
    #         closest_color_name = f"xkcd:{name}" # xkcd 색상은 접두사가 붙습니다.

    return closest_color_name.replace("gray", "grey").replace("grey", "gray")  # 일관성 위해 gray로 통일하거나 제거


# --- UI 요소 정의 및 설정 ---
# [UI 설정] 아이콘 크기 및 텍스트 설정
icon_base_size = 60  # 버튼 아이콘의 기본 너비/높이
button_text_scale = 0.9  # 버튼 텍스트 크기
button_text_thickness = 2  # 버튼 텍스트 두께

# 시뮬레이션 버튼 아이콘 (밝은 배경에 잘 보이도록 흰색 텍스트)
btn_icon_p = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_p[:] = (0, 0, 200)  # 빨간색
cv2.putText(btn_icon_p, "P", (int(icon_base_size * 0.25), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale, (255, 255, 255), button_text_thickness, cv2.LINE_AA)

btn_icon_d = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_d[:] = (0, 200, 0)  # 초록색
cv2.putText(btn_icon_d, "D", (int(icon_base_size * 0.25), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale, (255, 255, 255), button_text_thickness, cv2.LINE_AA)

btn_icon_t = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_t[:] = (200, 0, 0)  # 파란색
cv2.putText(btn_icon_t, "T", (int(icon_base_size * 0.25), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale, (255, 255, 255), button_text_thickness, cv2.LINE_AA)

# 보정 (Daltonize) 버튼 아이콘 (각 색상에 따라 텍스트 색상 조정)
btn_icon_pd = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_pd[:] = (0, 128, 255)  # 주황색 (BGR)
cv2.putText(btn_icon_pd, "PD", (int(icon_base_size * 0.15), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale * 0.8, (255, 255, 255), button_text_thickness, cv2.LINE_AA)

btn_icon_dd = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_dd[:] = (255, 255, 0)  # 청록색 (BGR)
cv2.putText(btn_icon_dd, "DD", (int(icon_base_size * 0.15), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale * 0.8, (0, 0, 0), button_text_thickness, cv2.LINE_AA)  # 검은색 텍스트

btn_icon_td = np.zeros((icon_base_size, icon_base_size, 3), dtype=np.uint8)
btn_icon_td[:] = (255, 0, 255)  # 자홍색 (BGR)
cv2.putText(btn_icon_td, "TD", (int(icon_base_size * 0.15), int(icon_base_size * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
            button_text_scale * 0.8, (255, 255, 255), button_text_thickness, cv2.LINE_AA)

# 버튼 활성화 시 오버레이할 색상
ACTIVE_OVERLAY_COLOR = (0, 255, 255)  # 청록색 (BGR)
ACTIVE_OVERLAY_ALPHA = 0.5  # 투명도

# [UI 설정] UI 요소 간의 간격 설정
padding = 15  # UI 요소의 가장자리 여백
item_spacing = 10  # UI 요소 간의 간격

# 현재 활성화된 모드 상태
active_cb_type = None  # "protan", "deutan", "tritan"
active_cb_mode = None  # "simulate", "correct"


# --- 마우스 클릭 처리를 위한 콜백 함수 ---
def handle_mouse_click(event, x, y, flags, param):
    global active_cb_type, active_cb_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        new_cb_type = None
        new_cb_mode = None

        # 버튼 영역 계산 (확대된 화면 기준으로 계산)
        # 시뮬레이션 버튼
        btn_sim_y_start = padding + 80  # 버튼 시작 Y 위치 (정보 패널 아래)
        btn_rects_sim = {
            "protan": {'x': padding, 'y': btn_sim_y_start, 'w': icon_base_size, 'h': icon_base_size},
            "deutan": {'x': padding + icon_base_size + item_spacing, 'y': btn_sim_y_start, 'w': icon_base_size,
                       'h': icon_base_size},
            "tritan": {'x': padding + 2 * (icon_base_size + item_spacing), 'y': btn_sim_y_start, 'w': icon_base_size,
                       'h': icon_base_size}
        }

        # 보정 버튼
        btn_corr_y_start = btn_sim_y_start + icon_base_size + item_spacing
        btn_rects_corr = {
            "protan": {'x': padding, 'y': btn_corr_y_start, 'w': icon_base_size, 'h': icon_base_size},
            "deutan": {'x': padding + icon_base_size + item_spacing, 'y': btn_corr_y_start, 'w': icon_base_size,
                       'h': icon_base_size},
            "tritan": {'x': padding + 2 * (icon_base_size + item_spacing), 'y': btn_corr_y_start, 'w': icon_base_size,
                       'h': icon_base_size}
        }

        # 시뮬레이션 버튼 클릭 확인
        for cb_type_key, rect in btn_rects_sim.items():
            if rect['x'] <= x <= rect['x'] + rect['w'] and rect['y'] <= y <= rect['y'] + rect['h']:
                if active_cb_type == cb_type_key and active_cb_mode == "simulate":
                    print(f"{cb_type_key.capitalize()} Simulation Disabled.")
                else:
                    new_cb_type = cb_type_key
                    new_cb_mode = "simulate"
                    print(f"{cb_type_key.capitalize()} Simulation Enabled.")
                break  # 이미 버튼을 찾았으므로 더 이상 확인하지 않음

        # 보정 버튼 클릭 확인 (시뮬레이션 버튼이 클릭되지 않았을 때만)
        if new_cb_type is None:
            for cb_type_key, rect in btn_rects_corr.items():
                if rect['x'] <= x <= rect['x'] + rect['w'] and rect['y'] <= y <= rect['y'] + rect['h']:
                    if active_cb_type == cb_type_key and active_cb_mode == "correct":
                        print(f"{cb_type_key.capitalize()} Correction Disabled.")
                    else:
                        new_cb_type = cb_type_key
                        new_cb_mode = "correct"
                        print(f"{cb_type_key.capitalize()} Correction Enabled.")
                    break

        # 실제 활성 모드 업데이트
        if new_cb_type == active_cb_type and new_cb_mode == active_cb_mode:
            active_cb_type = None
            active_cb_mode = None
        else:
            active_cb_type = new_cb_type
            active_cb_mode = new_cb_mode


# --- 비디오 처리 시작 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

window_title = "Color Detector & Color Blindness Simulation/Correction ('ESC': Exit)"
cv2.namedWindow(window_title)
cv2.setMouseCallback(window_title, handle_mouse_click)

cv2.createTrackbar("Severity (x100)", window_title, 100, 200, lambda x: None)

ret, initial_frame = cap.read()
if not ret:
    print("Could not read initial frame. Exiting.")
    exit()

original_h, original_w, _ = initial_frame.shape
scale_factor = 1.5  # 50% 확대 (원본 크기의 1.5배)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not receive frame (stream end?). Exiting...")
        break

    frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2

    b_orig, g_orig, r_orig = frame[center_y, center_x]
    rgb_at_center = (r_orig, g_orig, b_orig)
    color_name_at_center = get_color_name(rgb_at_center)

    severity_val = cv2.getTrackbarPos("Severity (x100)", window_title)
    severity = severity_val / 100.0

    processed_frame = frame.copy()
    if active_cb_type and active_cb_mode:
        if active_cb_mode == "simulate":
            processed_frame = simulate_colorblind_machado(frame, active_cb_type, severity=severity)
        elif active_cb_mode == "correct":
            processed_frame = daltonize(frame, active_cb_type, severity=severity, amplify=1.5)

    display_frame = processed_frame.copy()

    # --- UI 요소 그리기 ---
    # [UI 개선] 상단 정보 패널 (더 깔끔하게)
    panel_height = 70
    panel_width = w  # 화면 전체 너비 사용
    cv2.rectangle(display_frame, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)  # 검은색 배경
    cv2.rectangle(display_frame, (0, 0), (panel_width, panel_height), (50, 50, 50), 2)  # 테두리

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    main_font_scale = 0.8
    sub_font_scale = 0.6
    font_thickness = 2
    text_color_white = (255, 255, 255)
    text_color_yellow = (0, 255, 255)  # BGR 노란색

    # 모드 및 강도 정보 (우측 정렬)
    mode_display_text = "Mode: None"
    severity_display_text = ""
    if active_cb_type and active_cb_mode:
        mode_display_text = f"Mode: {active_cb_type.capitalize()} {active_cb_mode.capitalize()}"
        severity_display_text = f"Severity: {severity:.2f}"

    text_size_mode, _ = cv2.getTextSize(mode_display_text, font_face, sub_font_scale, font_thickness)
    text_size_severity, _ = cv2.getTextSize(severity_display_text, font_face, sub_font_scale, font_thickness)

    cv2.putText(display_frame, mode_display_text,
                (w - text_size_mode[0] - padding, 25), font_face, sub_font_scale, text_color_yellow, font_thickness,
                cv2.LINE_AA)
    cv2.putText(display_frame, severity_display_text,
                (w - text_size_severity[0] - padding, 55), font_face, sub_font_scale, text_color_yellow, font_thickness,
                cv2.LINE_AA)

    # RGB 정보 (왼쪽 정렬)
    # RGB 텍스트 크기 계산
    rgb_text = f"RGB: ({rgb_at_center[0]}, {rgb_at_center[1]}, {rgb_at_center[2]})"
    text_size_rgb, _ = cv2.getTextSize(rgb_text, font_face, main_font_scale, font_thickness)
    cv2.putText(display_frame, rgb_text,
                (padding, 35), font_face, main_font_scale, text_color_white, font_thickness, cv2.LINE_AA)

    # [UI 개선] 버튼 배치
    # 시뮬레이션 버튼 그룹
    btn_sim_y_start = panel_height + padding  # 정보 패널 아래
    btn_rects_sim = {
        "protan": {'x': padding, 'y': btn_sim_y_start, 'w': icon_base_size, 'h': icon_base_size, 'icon': btn_icon_p},
        "deutan": {'x': padding + icon_base_size + item_spacing, 'y': btn_sim_y_start, 'w': icon_base_size,
                   'h': icon_base_size, 'icon': btn_icon_d},
        "tritan": {'x': padding + 2 * (icon_base_size + item_spacing), 'y': btn_sim_y_start, 'w': icon_base_size,
                   'h': icon_base_size, 'icon': btn_icon_t}
    }
    # 보정 버튼 그룹
    btn_corr_y_start = btn_sim_y_start + icon_base_size + item_spacing
    btn_rects_corr = {
        "protan": {'x': padding, 'y': btn_corr_y_start, 'w': icon_base_size, 'h': icon_base_size, 'icon': btn_icon_pd},
        "deutan": {'x': padding + icon_base_size + item_spacing, 'y': btn_corr_y_start, 'w': icon_base_size,
                   'h': icon_base_size, 'icon': btn_icon_dd},
        "tritan": {'x': padding + 2 * (icon_base_size + item_spacing), 'y': btn_corr_y_start, 'w': icon_base_size,
                   'h': icon_base_size, 'icon': btn_icon_td}
    }

    # 버튼 그리기 및 활성화 상태 표시
    for cb_type_key, rect in btn_rects_sim.items():
        display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']] = rect['icon']
        if active_cb_type == cb_type_key and active_cb_mode == "simulate":
            overlay = display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']].copy()
            cv2.rectangle(overlay, (0, 0), (rect['w'], rect['h']), ACTIVE_OVERLAY_COLOR, -1)
            cv2.addWeighted(overlay, ACTIVE_OVERLAY_ALPHA,
                            display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']],
                            1 - ACTIVE_OVERLAY_ALPHA, 0,
                            display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']])
            cv2.rectangle(display_frame, (rect['x'], rect['y']), (rect['x'] + rect['w'], rect['y'] + rect['h']),
                          ACTIVE_OVERLAY_COLOR, 3)  # 테두리

    for cb_type_key, rect in btn_rects_corr.items():
        display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']] = rect['icon']
        if active_cb_type == cb_type_key and active_cb_mode == "correct":
            overlay = display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']].copy()
            cv2.rectangle(overlay, (0, 0), (rect['w'], rect['h']), ACTIVE_OVERLAY_COLOR, -1)
            cv2.addWeighted(overlay, ACTIVE_OVERLAY_ALPHA,
                            display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']],
                            1 - ACTIVE_OVERLAY_ALPHA, 0,
                            display_frame[rect['y']:rect['y'] + rect['h'], rect['x']:rect['x'] + rect['w']])
            cv2.rectangle(display_frame, (rect['x'], rect['y']), (rect['x'] + rect['w'], rect['y'] + rect['h']),
                          ACTIVE_OVERLAY_COLOR, 3)  # 테두리

    # 중앙 십자선
    cv2.drawMarker(display_frame, (center_x, center_y), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)

    # [UI 개선] 하단 색상 정보 및 스펙트럼 바
    spectrum_bar_height = 40  # 스펙트럼 바 높이 증가
    spectrum_bar_width = w - 2 * padding  # 화면 너비에 맞춤

    # 'Color: XXXXX' 텍스트 표시
    color_text = f"Color: {color_name_at_center}"
    text_size_color, _ = cv2.getTextSize(color_text, font_face, main_font_scale, font_thickness)

    # 텍스트가 스펙트럼 바 위에 위치하도록 계산
    # 스펙트럼 바의 Y 시작점
    spectrum_y_start = h - spectrum_bar_height
    # 텍스트의 Y 시작점 (스펙트럼 바 위에 여백을 두고)
    color_text_y_pos = spectrum_y_start - item_spacing - text_size_color[1]  # 텍스트 아래쪽 기준

    # 색상 이름 배경 박스
    cv2.rectangle(display_frame,
                  (padding - 5, color_text_y_pos - text_size_color[1] - 5),  # 좌상단 (패딩)
                  (padding + text_size_color[0] + 5, color_text_y_pos + 5),  # 우하단 (패딩)
                  (0, 0, 0), -1)  # 검은색 배경

    cv2.putText(display_frame, color_text,
                (padding, color_text_y_pos), font_face, main_font_scale, text_color_white, font_thickness, cv2.LINE_AA)

    # 감지된 색상 샘플 박스
    sample_box_size = 40
    sample_box_x = padding + text_size_color[0] + item_spacing + 20  # 색상 이름 텍스트 옆에 배치
    sample_box_y = color_text_y_pos - text_size_color[1]  # 텍스트의 상단과 맞춤
    cv2.rectangle(display_frame,
                  (sample_box_x, sample_box_y),
                  (sample_box_x + sample_box_size, sample_box_y + sample_box_size),
                  tuple(map(int, rgb_at_center[::-1])), -1)
    cv2.rectangle(display_frame,
                  (sample_box_x, sample_box_y),
                  (sample_box_x + sample_box_size, sample_box_y + sample_box_size),
                  (200, 200, 200), 1)  # 테두리

    # 스펙트럼 바 그리기
    spectrum = np.zeros((spectrum_bar_height, spectrum_bar_width, 3), dtype=np.uint8)
    for i in range(spectrum_bar_width):
        # HSV 색조 (0-179)를 BGR로 변환
        hsv_color = np.uint8([[[i * 180 // spectrum_bar_width, 255, 255]]])
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        spectrum[:, i] = bgr_color

    display_frame[spectrum_y_start:spectrum_y_start + spectrum_bar_height,
    padding:padding + spectrum_bar_width] = spectrum
    cv2.rectangle(display_frame,
                  (padding, spectrum_y_start),
                  (padding + spectrum_bar_width, spectrum_y_start + spectrum_bar_height),
                  (200, 200, 200), 1)  # 스펙트럼 바 테두리

    # 스펙트럼 막대에 하이라이트 그리기 (현재 색상 위치 표시)
    hsv_original_center_color = cv2.cvtColor(np.uint8([[rgb_at_center[::-1]]]), cv2.COLOR_BGR2HSV)[0][0]
    hue_val_from_cv = hsv_original_center_color[0]

    clipped_hue = np.clip(hue_val_from_cv, 0, 179)
    hue_pos_float = float(clipped_hue) * spectrum_bar_width / 180.0
    hue_pos_on_bar = int(hue_pos_float) + padding  # 스펙트럼 바의 시작 X 좌표 고려
    hue_pos_on_bar = max(padding, min(hue_pos_on_bar, w - padding - 1))  # 화면 경계 내로 제한

    cv2.line(display_frame, (hue_pos_on_bar, spectrum_y_start),
             (hue_pos_on_bar, spectrum_y_start + spectrum_bar_height), (255, 255, 255), 2)  # 하이라이트 라인

    # 5. 최종 프레임 표시
    cv2.imshow(window_title, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()