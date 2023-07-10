import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image

# 데이터셋 클래스 정의
class SatelliteDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        image = np.load(file_path)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로드
data_dir = "인공위성_데이터_폴더_경로"
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
dataset = SatelliteDataset(data_dir=data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# U-Net 모델 정의
model = fcn_resnet50(pretrained=False, num_classes=2)
model.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for images in data_loader:
        images = images.to(device)

        optimizer.zero_grad()

        # Forward 패스
        outputs = model(images)['out']

        # 가짜 마스크 생성 (임시로 사용)
        masks = torch.zeros_like(outputs)
        masks[:, 1, :, :] = 1  # 건물 클래스에 해당하는 채널에 1로 설정

        # 손실 계산
        loss = criterion(outputs, masks.argmax(dim=1))

        # 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 에포크별 학습 손실 출력
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(data_loader):.4f}")

# 학습된 모델 저장
torch.save(model.state_dict(), "모델_저장_경로")
