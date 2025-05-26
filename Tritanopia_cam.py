import cv2
import numpy as np

# Machado (2009) 변환 행렬 (severity=1.0 기준)
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
    matrix = CVD_MATRICES[cb_type]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    reshaped = img_rgb.reshape(-1, 3)

    simulated = reshaped @ matrix.T
    simulated = (1 - severity) * reshaped + severity * simulated
    simulated = np.clip(simulated, 0, 1)

    error = reshaped - simulated
    corrected = reshaped + amplify * error
    corrected = np.clip(corrected, 0, 1)

    result = (corrected.reshape(img.shape) * 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

def nothing(x): pass

# 모드 목록
mode_text = [
    "Original",
    "Protanopia Simulation",
    "Protanopia Correction",
    "Deuteranopia Simulation",
    "Deuteranopia Correction",
    "Tritanopia Simulation",
    "Tritanopia Correction"
]
mode_map = {
    0: ("original", None),
    1: ("simulate", "protan"),
    2: ("correct", "protan"),
    3: ("simulate", "deutan"),
    4: ("correct", "deutan"),
    5: ("simulate", "tritan"),
    6: ("correct", "tritan"),
}

# 웹캠 설정
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("웹캠을 열 수 없습니다.")

cv2.namedWindow("Colorblind Viewer")
cv2.createTrackbar("Mode", "Colorblind Viewer", 0, 6, nothing)
cv2.createTrackbar("Severity x100", "Colorblind Viewer", 100, 200, nothing)  # 기본값: 1.0

print("🎥 색약 시뮬레이션 + 보정 시작 (ESC로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    mode = cv2.getTrackbarPos("Mode", "Colorblind Viewer")
    severity_val = cv2.getTrackbarPos("Severity x100", "Colorblind Viewer")
    severity = severity_val / 100.0  # 트랙바 값 0~200 → 0.0 ~ 2.0

    func_type, cb_type = mode_map[mode]

    if func_type == "original":
        output = frame
    elif func_type == "simulate":
        output = simulate_colorblind_machado(frame, cb_type, severity=severity)
    elif func_type == "correct":
        output = daltonize(frame, cb_type, severity=severity)
    else:
        output = frame

    cv2.putText(output, f"Mode: {mode_text[mode]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(output, f"Severity: {severity:.2f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Colorblind Viewer", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
