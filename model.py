import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 색상 보정 CNN 모델
class ColorCorrectionCNN(nn.Module):
    def __init__(self):
        super(ColorCorrectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 3 * 128 * 128)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 3, 128, 128)
        return x

# 프레임 보정 함수
def correct_frame(frame, model):
    frame = cv2.resize(frame, (128, 128))
    tensor = torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    result = output.squeeze().permute(1, 2, 0).numpy() * 255
    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.resize(result, (640, 480))  # 원래 크기로 복원
    return result

# 메인 루프
def run_video_correction():
    cap = cv2.VideoCapture(0)  # 웹캠 사용 (0번 장치)
    model = ColorCorrectionCNN()
    model.load_state_dict(torch.load("color_correction_model.pth"))
    model.eval()

    print("실시간 색맹 보정 테스트 시작 (q 누르면 종료)")

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
    run_video_correction()
