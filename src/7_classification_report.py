import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

BASE = r"D:\sem_4\Project\Maize_Leaf_Project"
TEST_DIR = BASE + r"\split_data\test"
MODEL_PATH = BASE + r"\models\resnet18.pth"

CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

y_true, y_pred = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = model(imgs)
        preds = torch.argmax(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print("\n Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASSES))
