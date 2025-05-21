import cv2
import torch
import torch.nn.functional as F
from model import ColorCorrectionCNN  # 네가 만든 model.py 파일에서 불러오기

# 실시간 프레임 보정 함수
def correct_frame(frame, model):
    # 1. 프레임 전처리
    frame_resized = cv2.resize(frame, (128, 128))
    tensor = torch.tensor(frame_resized / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # 2. 모델 추론
    with torch.no_grad():
        output = model(tensor)

    # 3. 후처리
    result = output.squeeze().permute(1, 2, 0).numpy()
    result = (result * 255).clip(0, 255).astype("uint8")
    result = cv2.resize(result, (640, 480))  # 원본 해상도로 복원
    return result

# 메인 함수
def run_realtime_correction():
    cap = cv2.VideoCapture(0)  # 0번 카메라 (웹캠)
    model = ColorCorrectionCNN()

    # 학습된 가중치 불러오기
    model.load_state_dict(torch.load("color_correction_model.pth", map_location='cpu'))
    model.eval()

    print("✅ 실시간 색맹 보정 시작 (q 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected = correct_frame(frame, model)

        cv2.imshow("원본 영상", frame)
        cv2.imshow("보정된 영상", corrected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    run_realtime_correction()
