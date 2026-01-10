import os, shutil, random

base_dir = r"D:\sem_4\Project\Maize_Leaf_Project\data"
out_dir  = r"D:\sem_4\Project\Maize_Leaf_Project\split_data"

train_r, val_r = 0.7, 0.15

for split in ["train", "val", "test"]:
    for cls in os.listdir(base_dir):
        os.makedirs(os.path.join(out_dir, split, cls), exist_ok=True)

for cls in os.listdir(base_dir):
    imgs = os.listdir(os.path.join(base_dir, cls))
    random.shuffle(imgs)

    n = len(imgs)
    t = int(n * train_r)
    v = int(n * val_r)

    for img in imgs[:t]:
        shutil.copy(os.path.join(base_dir, cls, img),
                    os.path.join(out_dir, "train", cls, img))

    for img in imgs[t:t+v]:
        shutil.copy(os.path.join(base_dir, cls, img),
                    os.path.join(out_dir, "val", cls, img))

    for img in imgs[t+v:]:
        shutil.copy(os.path.join(base_dir, cls, img),
                    os.path.join(out_dir, "test", cls, img))

print("Train–Val–Test split completed")
