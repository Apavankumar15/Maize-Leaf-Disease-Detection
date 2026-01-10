
print(" Training started...")

import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

classes = ['Blight','Common_Rust','Gray_Leaf_Spot','Healthy']

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(
    r"D:\sem_4\Project\Maize_Leaf_Project\models\resnet18.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

img = Image.open(r"D:\sem_4\Project\Maize_Leaf_Project\test_leaf.jpg")
img = transform(img).unsqueeze(0)

with torch.no_grad():
    pred = model(img).argmax(1).item()

print(" Predicted Disease:", classes[pred])
