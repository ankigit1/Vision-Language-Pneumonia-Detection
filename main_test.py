import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utils.dataset_loader import CXRDataset
from models.siamese_densenet_lbp import SiameseDenseLBP

test_data = CXRDataset("data/test")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseDenseLBP().to(device)
model.load_state_dict(torch.load("checkpoints/pneumonia_siamese_vlm_lambda0.25.pt"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = torch.argmax(model(imgs), dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix on Test Data")
plt.savefig("results/confusion_matrices/test_confusion.png")
plt.show()
