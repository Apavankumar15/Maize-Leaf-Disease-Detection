from torchvision import datasets, transforms
from torch.utils.data import DataLoader

base = r"D:\sem_4\Project\Maize_Leaf_Project\split_data"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train = datasets.ImageFolder(base+r"\train", transform=transform)
val   = datasets.ImageFolder(base+r"\val", transform=transform)
test  = datasets.ImageFolder(base+r"\test", transform=transform)

print("Classes:", train.classes)
print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
