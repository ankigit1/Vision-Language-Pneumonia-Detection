import torch
from torch.utils.data import DataLoader
from models.siamese_densenet_lbp import SiameseDenseLBP
from utils.dataset_loader import CXRDataset
from utils.training_utils import train_epoch, evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_data = CXRDataset("data/train")
val_data = CXRDataset("data/val")

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

model = SiameseDenseLBP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    loss = train_epoch(model, train_loader, optimizer, device)
    metrics = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={metrics['acc']:.2f}, F1={metrics['f1']:.2f}")

torch.save(model.state_dict(), "checkpoints/pneumonia_siamese_vlm_lambda0.25.pt")
