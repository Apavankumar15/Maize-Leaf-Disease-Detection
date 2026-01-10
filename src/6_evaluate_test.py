import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# CONFIG
# ------------------------
BASE = r"D:\sem_4\Project\Maize_Leaf_Project"
TEST_DIR = BASE + r"\split_data\test"
MODEL_PATH = BASE + r"\models\resnet18.pth"

CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
BATCH_SIZE = 32

# ------------------------
# TRANSFORMS
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------
# DATASET & LOADER
# ------------------------
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------------------
# LOAD MODEL
# ------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ------------------------
# TESTING
# ------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# ------------------------
# ACCURACY
# ------------------------
test_acc = accuracy_score(all_labels, all_preds) * 100
print(f"\n✅ Test Accuracy: {test_acc:.2f}%")

# ------------------------
# CONFUSION MATRIX
# ------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Maize Leaf Disease")
plt.show()
