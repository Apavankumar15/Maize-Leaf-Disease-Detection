print("Training started...")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ------------------------
# PATHS
# ------------------------
BASE = r"D:\sem_4\Project\Maize_Leaf_Project"
TRAIN_DIR = BASE + r"\split_data\train"
VAL_DIR   = BASE + r"\split_data\val"
MODEL_DIR = BASE + r"\models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------
# TRANSFORMS
# ------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ------------------------
# DATA LOADERS
# ------------------------
train_data = datasets.ImageFolder(TRAIN_DIR, transform)
val_data   = datasets.ImageFolder(VAL_DIR, transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32)

# ------------------------
# MODEL
# ------------------------
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)

# ------------------------
# LOSS & OPTIMIZER
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ------------------------
# TRAINING
# ------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    correct = total = 0

    for imgs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {acc:.2f}%")

# ------------------------
# SAVE MODEL  (MOST IMPORTANT)
# ------------------------
model_path = MODEL_DIR + r"\resnet18.pth"
torch.save(model.state_dict(), model_path)

print("TRAINING COMPLETE")
print("Model saved at:", model_path)
