import os
from PIL import Image

dataset_path = r"D:\sem_4\Project\Maize_Leaf_Project\data"

corrupted = 0

for cls in os.listdir(dataset_path):
    cls_path = os.path.join(dataset_path, cls)
    if not os.path.isdir(cls_path):
        continue

    for img in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img)
        try:
            with Image.open(img_path) as im:
                im.verify()
        except:
            os.remove(img_path)
            corrupted += 1

print(" Cleaning completed")
print(" Corrupted images removed:", corrupted)
