from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as T

class ColorCorrectionDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform or T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])
        input_img = self.transform(Image.open(input_path).convert("RGB"))
        target_img = self.transform(Image.open(target_path).convert("RGB"))
        return input_img, target_img
