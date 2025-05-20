import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ColorCorrectionCNN  # 앞서 만든 모델
from data_set import ColorCorrectionDataset

# 데이터 경로 설정
input_path = 'dataset/simulated_cb'
target_path = 'dataset/normal'

# 데이터셋, DataLoader 구성
dataset = ColorCorrectionDataset(input_path, target_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 모델, 손실함수, 옵티마이저
model = ColorCorrectionCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
epochs = 20
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

# 모델 저장
torch.save(model.state_dict(), "color_correction_model.pth")
print("✅ 모델 저장 완료!")
