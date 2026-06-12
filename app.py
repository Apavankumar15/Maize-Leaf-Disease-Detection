import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Title
st.title("🌽 Maize Leaf Disease Detection")
st.write("Upload a maize leaf image and predict the disease.")

# Classes
classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(
        torch.load("models/resnet18.pth", map_location=torch.device("cpu"))
    )

    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Upload image
uploaded_file = st.file_uploader(
    "Choose a maize leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)

        probs = F.softmax(outputs, dim=1)[0]

        pred = torch.argmax(probs).item()

    st.success(
        f"Predicted Disease: {classes[pred]}"
    )

    st.subheader("Confidence Scores")

    for i, cls in enumerate(classes):
        st.write(
            f"{cls}: {probs[i].item()*100:.2f}%"
        )