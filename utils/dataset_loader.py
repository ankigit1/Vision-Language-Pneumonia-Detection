from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class CXRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.classes = sorted(os.listdir(root_dir))
        self.paths, self.labels = [], []
        for i, c in enumerate(self.classes):
            for f in os.listdir(os.path.join(root_dir, c)):
                self.paths.append(os.path.join(root_dir, c, f))
                self.labels.append(i)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]
