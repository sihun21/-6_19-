import cv2
import numpy as np
import matplotlib.colors as mcolors

# 색약 시뮬레이션 변환 행렬
CVD_MATRICES = {
    "protan": np.array([
        [0.630323, 0.465641, -0.095964],
        [0.069181, 0.890046, 0.040773],
        [-0.006308, -0.007724, 1.014032]
    ]),
    "deutan": np.array([
        [0.675425, 0.433850, -0.109275],
        [0.125303, 0.847755, 0.026942],
        [-0.007950, 0.018572, 0.989378]
    ]),
    "tritan": np.array([
        [1.104996, -0.046633, -0.058363],
        [-0.032137, 0.971635, 0.060503],
        [0.001336, 0.317922, 0.680742]
    ])
}

def simulate(img, cb_type, severity=1.0):
    if cb_type not in CVD_MATRICES:
        return img
    mat = CVD_MATRICES[cb_type]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    reshaped = rgb.reshape(-1, 3)
    transformed = reshaped @ mat.T
    blended = (1 - severity) * reshaped + severity * np.clip(transformed, 0, 1)
    blended = (blended.reshape(img.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

def daltonize(img, cb_type, severity=1.0, amplify=1.5):
    if cb_type not in CVD_MATRICES:
        return img
    mat = CVD_MATRICES[cb_type]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    reshaped = rgb.reshape(-1, 3)
    simulated = reshaped @ mat.T
    blended = (1 - severity) * reshaped + severity * np.clip(simulated, 0, 1)
    error = reshaped - blended
    corrected = np.clip(reshaped + amplify * error, 0, 1)
    corrected = (corrected.reshape(img.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

def create_red_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 160, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 160, 60])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
    red_mask = red_mask.astype(np.float32) / 255.0
    return np.expand_dims(red_mask, axis=2)

def simulate_colorblind_machado(img, cb_type, severity=1.0):
    matrix = CVD_MATRICES.get(cb_type)
    if matrix is None:
        return img
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = img_rgb.reshape(-1, 3)
    simulated = reshaped @ matrix.T
    blended = (1 - severity) * reshaped + severity * np.clip(simulated, 0, 1)
    return cv2.cvtColor((blended.reshape(img.shape) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def apply_red_mask_simulation(img_bgr, cb_type="protan", severity=1.0):
    red_mask = create_red_mask(img_bgr)
    simulated_frame = simulate_colorblind_machado(img_bgr, cb_type, severity)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    simulated_rgb = cv2.cvtColor(simulated_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended_img = red_mask * simulated_rgb + (1 - red_mask) * img_rgb
    blended_img = np.clip(blended_img, 0, 1)
    return cv2.cvtColor((blended_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def red_mask_daltonize(img_bgr, severity=1.0, amplify=1.5):
    protan_matrix = CVD_MATRICES["protan"]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = img_rgb.reshape(-1, 3)
    simulated = reshaped @ protan_matrix.T
    simulated = (1 - severity) * reshaped + severity * simulated
    error = reshaped - simulated
    corrected = reshaped + amplify * error
    corrected = np.clip(corrected, 0, 1).reshape(img_rgb.shape)
    red_mask = create_red_mask(img_bgr)
    result = red_mask * corrected + (1 - red_mask) * img_rgb
    result = np.clip(result, 0, 1)
    return cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

def get_color_name(rgb_tuple):
    rgb_norm = tuple(c / 255.0 for c in rgb_tuple)
    name = "Unknown"
    min_dist = float('inf')
    for n, hex_val in mcolors.CSS4_COLORS.items():
        ref = mcolors.to_rgb(hex_val)
        dist = sum((a - b) ** 2 for a, b in zip(rgb_norm, ref))
        if dist < min_dist:
            min_dist, name = dist, n
    return name.replace("grey", "gray")

# --------------------------
# UI 관련 변수
icon_size = 60
padding = 15
item_spacing = 10
current_hue = None

# 버튼 이미지 정의 (초록/파랑 제거)
button_imgs = {
    "protan_sim": np.full((icon_size, icon_size, 3), (0, 0, 255), np.uint8),
    "protan_corr": np.full((icon_size, icon_size, 3), (128, 128, 255), np.uint8),
    "rd_corr": np.full((icon_size, icon_size, 3), (0, 0, 128), np.uint8),
    "rs_sim": np.full((icon_size, icon_size, 3), (0, 255, 255), np.uint8)
}
cv2.putText(button_imgs["rd_corr"], "RD", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(button_imgs["rs_sim"], "RS", (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

# 시뮬레이션 버튼 위치 정의 (protan만 유지)
btns_sim = {
    "protan": (padding, padding + 60)
}

# 보정 버튼 위치 정의 (protan과 red 계열만 유지)
btns_corr = {
    "protan": (padding, padding + 60),
    "rd": (padding + (icon_size + item_spacing), padding + 60),
    "rs": (padding + 2 * (icon_size + item_spacing), padding + 60)
}

# draw loop 내 버튼 그리기에서 불필요 항목 제거할 필요는 없음
# 이미 위에서 정의된 key만을 대상으로 하므로 안전


mouse_x, mouse_y = -1, -1
left_mode, right_mode = (None, None), (None, None)

def mouse_cb(event, x, y, flags, param):
    global mouse_x, mouse_y, left_mode, right_mode
    mouse_x, mouse_y = x, y
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    full_w, _ = param
    lw = full_w // 2
    if x < lw:
        for typ, (bx, by) in btns_sim.items():
            if bx <= x <= bx + icon_size and by <= y <= by + icon_size:
                left_mode = (typ, "simulate") if left_mode != (typ, "simulate") else (None, None)
    else:
        rx = x - lw
        for typ, (bx, by) in btns_corr.items():
            if bx <= rx <= bx + icon_size and by <= y <= by + icon_size:
                target = ("protan", "red_correct") if typ == "rd" else ("protan", "red_simulate") if typ == "rs" else (typ, "correct")
                right_mode = target if right_mode != target else (None, None)

def draw_ui(img, mode, mouse_x, mouse_y, offset_x):
    global current_hue
    h, w = img.shape[:2]
    panel_h = 60
    cv2.rectangle(img, (0, 0), (w, panel_h), (0, 0, 0), -1)

    px = mouse_x - offset_x
    if 0 <= px < w and 0 <= mouse_y < h:
        bgr = img[mouse_y, px]
        rgb = tuple(int(c) for c in bgr[::-1])
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        current_hue = int(hsv[0])  # hue 값 설정
        name = get_color_name(rgb)
        cv2.putText(img, f"RGB: {rgb}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(img, f"Color: {name}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.drawMarker(img, (px, mouse_y), (200, 200, 200), cv2.MARKER_CROSS, 20, 1)
    else:
        current_hue = None  # 유효하지 않은 위치면 hue 리셋

    cv2.putText(img, f"Mode: {mode[0] if mode[0] else 'None'}", (w - 220, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    sev = cv2.getTrackbarPos("Severity", window_title) / 100.0
    cv2.putText(img, f"Severity: {sev:.2f}", (w - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def draw_spectrum(img, hue):
    h, w = img.shape[:2]
    sw = w - 2 * padding
    spectrum = np.zeros((40, sw, 3), dtype=np.uint8)

    for i in range(sw):
        hsv = np.array([[[i * 180 / sw, 255, 255]]], dtype=np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        spectrum[:, i] = bgr

    yb = h - 40 - padding
    img[yb:yb+40, padding:padding+sw] = spectrum
    cv2.rectangle(img, (padding, yb), (padding+sw, yb+40), (200, 200, 200), 1)

    # hue가 None이 아니고, 정수 변환 가능한 값일 때만 포인터 표시
    if hue is not None and isinstance(hue, (int, float)):
        pos = int(hue * sw / 180)
        pos = np.clip(pos, 0, sw - 1)
        cv2.line(img, (padding + pos, yb), (padding + pos, yb + 40), (255, 255, 255), 2)

# -------- Main 실행 --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Camera open failed")

window_title = "Color Blindness Simulator with RD+RS"
cv2.namedWindow(window_title)
cv2.createTrackbar("Severity", window_title, 100, 200, lambda x: None)
ret, frame = cap.read()
if not ret:
    raise Exception("Camera read failed")
cv2.setMouseCallback(window_title, mouse_cb, param=(frame.shape[1] * 2, frame.shape[0]))

while True:

    ret, frame = cap.read()
    if not ret:
        break
    sev = cv2.getTrackbarPos("Severity", window_title) / 100.0
    h, w = frame.shape[:2]
    left = simulate(frame.copy(), left_mode[0], sev) if left_mode[1] == "simulate" else frame.copy()
    if right_mode[1] == "correct":
        right = daltonize(frame.copy(), right_mode[0], sev)
    elif right_mode[1] == "red_correct":
        right = red_mask_daltonize(frame.copy(), severity=sev)
    elif right_mode[1] == "red_simulate":
        right = apply_red_mask_simulation(frame.copy(), cb_type="protan", severity=sev)
    else:
        right = frame.copy()
    px = mouse_x
    py = mouse_y
    if 0 <= py < h:
        if 0 <= px < w:
            bgr = left[py, px]
        elif w <= px < 2 * w:
            bgr = right[py, px - w]
        else:
            bgr = None
        if bgr is not None:
            hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            current_hue = int(hsv[0])
        else:
            current_hue = None
    else:
        current_hue = None

    for typ, (x, y) in btns_sim.items():
        key = f"{typ}_sim"
        if key in button_imgs:
            left[y:y+icon_size, x:x+icon_size] = button_imgs[key]

    for typ, (x, y) in btns_corr.items():
        key = "rd_corr" if typ == "rd" else "rs_sim" if typ == "rs" else f"{typ}_corr"
        if key in button_imgs:
            right[y:y+icon_size, x:x+icon_size] = button_imgs[key]

    draw_ui(left, left_mode, mouse_x, mouse_y, 0)
    draw_ui(right, right_mode, mouse_x, mouse_y, w)
    draw_spectrum(left, current_hue)
    draw_spectrum(right, current_hue)

    combined = np.hstack((left, right))
    cv2.imshow(window_title, combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
